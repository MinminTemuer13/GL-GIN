import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 模块 0: RMSNorm 实现 ---
class RMSNorm(nn.Module):
    """
    均方根层归一化 (Root Mean Square Layer Normalization)。
    论文中提到使用 RMSNorm。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        初始化 RMSNorm。

        参数:
            dim (int): 输入特征的维度。
            eps (float): 为防止除以零而加入的小值。
        """
        super().__init__()
        self.eps = eps
        #可学习的缩放参数 gamma
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """计算 RMSNorm """
        # x * 1/sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        输入的形状是 (..., dim)
        """
        output = self._norm(x.float()).type_as(x) # 转换为float计算，然后转回原类型
        return output * self.weight


# --- 模块 1: 旋转位置编码 (RotaryEmbedding) ---
class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE) 模块。
    它通过旋转Query和Key向量的某些维度来注入相对位置信息。
    """
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0, base: int = 10000):
        """
        初始化RoPE。

        参数:
            dim (int): RoPE应用的特征维度 (通常是每个注意力头的维度 d_head)。
                       必须是偶数。
            max_seq_len (int): 模型能处理的最大序列长度，用于预计算频率。
            theta (float): RoPE中的基础周期参数，与论文中的theta一致。
                           (注意：在一些实现中，这个参数可能被称为 `base`)
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE的维度 'dim' ({dim}) 必须是偶数。")

        # 计算频率，与Transformer的PositionalEncoding类似但用于旋转
        # freqs 的形状是 (dim / 2)
        # θ_i = base^(-2i/dim)
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

        # t 代表位置索引 m，形状是 (max_seq_len)
        t = torch.arange(max_seq_len, device=freqs.device)
        # freqs_for_rotation (m * θ_i) 的形状是 (max_seq_len, dim / 2)
        freqs_for_rotation = torch.outer(t, freqs)

        # 缓存cos和sin值，形状 (1, max_seq_len, 1, dim / 2) 以便广播
        # (bs, seq_len, n_heads, head_dim) -> RoPE作用于head_dim
        # 如果Q/K是 (bs, n_heads, seq_len, head_dim)，那么cos/sin缓存需要匹配
        # 我们在注意力模块中Q/K的形状是 (bs, n_heads, seq_len, head_dim)
        # RoPE希望作用于最后一个维度，所以cos/sin需要是 (1, seq_len, 1, dim/2)
        # 或者在应用时调整Q/K的维度顺序。
        # 更通用的做法是让cos/sin的seq_len维度在前。
        # (max_seq_len, dim // 2)
        cos_cached = torch.cos(freqs_for_rotation)
        sin_cached = torch.sin(freqs_for_rotation)

        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        self.dim = dim

    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        对输入x应用旋转。

        参数:
            x (torch.Tensor): 输入张量, 形状 [B, H, S, D_h] 或 [B, S, D_model] (如果作用于整个d_model)。
                              这里的D_h或D_model是RoPE作用的维度（即self.dim）。
            cos (torch.Tensor): 对应序列长度的cos项, 形状 [S, D_h/2] 或 [S, D_model/2]。
            sin (torch.Tensor): 对应序列长度的sin项, 形状 [S, D_h/2] 或 [S, D_model/2]。
        返回:
            torch.Tensor: 旋转后的张量。
        """
        # 将x的最后一维拆分为两半: x_even, x_odd
        # x_even: x[..., 0:dim:2]
        # x_odd:  x[..., 1:dim:2]
        # 另一种实现方式：
        # x_rope = x[..., :self.dim]
        # x_pass = x[..., self.dim:] (如果只对部分维度应用RoPE)

        # 这里我们假设x的最后一维完全用于RoPE (dim == x.shape[-1])
        x_part1 = x[..., : self.dim // 2]  # (B, H, S, D_h/2)
        x_part2 = x[..., self.dim // 2 :]  # (B, H, S, D_h/2)

        # 调整cos和sin的形状以匹配x_part1/x_part2进行广播
        # cos/sin: [S, D_h/2] -> [1, 1, S, D_h/2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # 应用旋转公式:
        # x_rot_part1 = x_part1 * cos - x_part2 * sin
        # x_rot_part2 = x_part1 * sin + x_part2 * cos
        rotated_x_part1 = x_part1 * cos - x_part2 * sin
        rotated_x_part2 = x_part1 * sin + x_part2 * cos

        return torch.cat((rotated_x_part1, rotated_x_part2), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量x (通常是Q或K的多头表示) 应用旋转位置编码。

        参数:
            x (torch.Tensor): 输入张量，预期形状: [batch_size, num_heads, seq_len, head_dim]。
                              RoPE将作用于最后一个维度 (head_dim)。
        返回:
            torch.Tensor: 应用RoPE后的张量，形状与x相同。
        """
        # x: [B, H, S, D_h]
        seq_len = x.shape[2] # S (序列长度)
        # 取出对应长度的cos和sin缓存
        cos = self.cos_cached[:seq_len] # [S, D_h/2]
        sin = self.sin_cached[:seq_len] # [S, D_h/2]

        return self._apply_rotary_emb(x, cos, sin)


# --- 模块 2: 邻接亲和度计算器 (NeighborAffinityCalculator) ---
class NeighborAffinityCalculator(nn.Module):
    """
    根据论文公式 (3), (4), (5) 计算邻接亲和度分数 a_i,i+1。

    这个模块负责计算相邻词元之间的“合并”或“连接”倾向。
    """
    def __init__(self, d_model: int, d_k_neighbor: int):
        """
        初始化邻接亲和度计算器。

        参数:
            d_model (int): 模型的维度。
            d_k_neighbor (int): 论文中 s_i,i+1 公式中的缩放因子 d_s。
        """
        super().__init__()
        self.d_k_neighbor = d_k_neighbor
        # 用于计算亲和度分数 s_i,i+1 的 W_Q 和 W_K 矩阵
        self.wq_neighbor = nn.Linear(d_model, d_model)
        self.wk_neighbor = nn.Linear(d_model, d_model)

    def forward(self, x_input_for_affinity: torch.Tensor,
                prev_affinity_scores_a: torch.Tensor | None = None,
                layer_idx: int = 0) -> torch.Tensor | None:
        """
        计算当前层的邻接亲和度分数。

        参数:
            x_input_for_affinity (torch.Tensor): 当前层用于计算亲和度的输入，形状 [batch_size, seq_len, d_model]。
            prev_affinity_scores_a (torch.Tensor | None): 上一层计算得到的 'a' 分数，形状 [batch_size, seq_len-1]。
                                                         对于第一层，此值为 None。
            layer_idx (int): 当前层的索引 (0-indexed)。

        返回:
            torch.Tensor | None: 更新后的邻接亲和度分数 'a'，形状 [batch_size, seq_len-1]。
                                 如果序列长度 <= 1，则返回 None。
        """
        batch_size, seq_len, _ = x_input_for_affinity.shape
        if seq_len <= 1: # 序列长度不足以计算邻接分数
            return None

        # --- 公式 (3): 计算邻接注意力分数 s_i,i+1 ---
        # q_for_s 对应 x_i W^Q，选择除了最后一个token外的所有token
        q_for_s = self.wq_neighbor(x_input_for_affinity[:, :-1, :])  # [B, S-1, D]
        # k_for_s 对应 x_{i+1} W^K，选择除了第一个token外的所有token
        k_for_s = self.wk_neighbor(x_input_for_affinity[:, 1:, :])   # [B, S-1, D]
        # 计算点积并缩放得到 s_forward (s_i,i+1)
        s_forward = torch.sum(q_for_s * k_for_s, dim=-1) / self.d_k_neighbor  # [B, S-1]

        # 计算 s_{i+1,i} (反向亲和力)
        q_for_s_rev = self.wq_neighbor(x_input_for_affinity[:, 1:, :])
        k_for_s_rev = self.wk_neighbor(x_input_for_affinity[:, :-1, :])
        s_backward = torch.sum(q_for_s_rev * k_for_s_rev, dim=-1) / self.d_k_neighbor # [B, S-1]

        # --- 公式 (4): 计算邻接亲和度初步分数 â_i,i+1 ---
        # 论文中是 (softmax(s_i,i+1) + softmax(s_i+1,i)) / 2。
        # 此处使用sigmoid作为softmax的简化替代，将分数映射到(0,1)区间，代表一种“强度”。
        affinity_hat_forward = torch.sigmoid(s_forward)
        affinity_hat_backward = torch.sigmoid(s_backward)
        current_layer_affinity_hat = (affinity_hat_forward + affinity_hat_backward) / 2 # â_i,i+1, [B, S-1]

        # --- 公式 (5): 层级更新亲和度分数 a^l_i,i+1 ---
        if layer_idx == 0: # l=0 (第一层) 的情况
            updated_affinity_scores_a = current_layer_affinity_hat
        else: # l>=1 (后续层) 的情况
            if prev_affinity_scores_a is None: # 理论上对于seq_len > 1且layer_idx > 0时，此项不应为None
                 raise ValueError("当 layer_idx > 0 且 seq_len > 1 时, prev_affinity_scores_a 不能为空。")
            # a^{l}_{i,i+1} = a^{l-1}_{i,i+1} + (1 - a^{l-1}_{i,i+1}) * â^l_{i,i+1}
            updated_affinity_scores_a = prev_affinity_scores_a + \
                                     (1 - prev_affinity_scores_a) * current_layer_affinity_hat
        return updated_affinity_scores_a


# --- 模块 3: 构建层级注意力掩码 (辅助函数) ---
def build_hierarchical_attention_mask(affinity_scores_a: torch.Tensor | None,
                                      seq_len: int,
                                      device: torch.device) -> torch.Tensor:
    """
    根据论文公式 (6) 构建层级注意力掩码 C。

    这个掩码 C_ij 表示从位置 i 到位置 j 的路径上的累积亲和度。

    参数:
        affinity_scores_a (torch.Tensor | None): 当前层计算得到的 'a' 分数，形状 [batch_size, seq_len-1]。
                                                 如果序列长度 <= 1 (即affinity_scores_a为None)，则返回允许所有注意力的掩码。
        seq_len (int): 序列长度。
        device (torch.device): 计算设备 (例如 'cpu' 或 'cuda')。

    返回:
        torch.Tensor: 层级注意力掩码 C，形状 [batch_size, 1, seq_len, seq_len]，
                      准备好用于多头注意力的广播。
    """
    if affinity_scores_a is None: # 如果序列长度 <= 1，或无法计算亲和度
        # 返回一个允许所有token互相注意的掩码 (尽管单token序列的自注意力意义不大)
        # 或者如果seq_len > 1 但 affinity_scores_a 为 None (不应发生), 也允许所有注意
        return torch.ones(1, 1, seq_len, seq_len, device=device, dtype=torch.float)

    batch_size = affinity_scores_a.shape[0]
    # 初始化 C_i,j = 1 (当 i=j 时)，即对角线为1
    C = torch.eye(seq_len, device=device, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1, 1) # [B, S, S]

    # 计算 C_i,j 当 i < j 时: Π_{k=i}^{j-1} a_{k,k+1}
    # affinity_scores_a[:, k] 对应论文中的 a_{k,k+1}
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            # 路径上的亲和度序列: a_{i,i+1}, a_{i+1,i+2}, ..., a_{j-1,j}
            # 对应 affinity_scores_a 的索引是 i, i+1, ..., j-1
            if j - 1 >= i: # 确保路径存在 (即 j > i)
                path_affinities = affinity_scores_a[:, i:j] # 切片得到 [batch_size, j-i]
                C[:, i, j] = torch.prod(path_affinities, dim=1) # 沿路径累乘
                C[:, j, i] = C[:, i, j] # 掩码是对称的 C_ij = C_ji

    # 增加一个head维度以便广播: [batch_size, 1, seq_len, seq_len]
    return C.unsqueeze(1)

# --- 模块 4: 缩放点积注意力 (基础单元) ---
class ScaledDotProductAttention(nn.Module):
    """计算缩放点积注意力权重。"""
    def __init__(self, dropout_rate: float = 0.0): # 通常在MHA级别或子层级别应用dropout
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) # 可选的dropout，DIFF论文未明确在此处用

    def forward(self, q_heads: torch.Tensor, k_heads: torch.Tensor,
                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        参数:
            q_heads (torch.Tensor): Query张量, 形状 [B, H, S_q, D_h]。
            k_heads (torch.Tensor): Key张量, 形状 [B, H, S_k, D_h]。
            padding_mask (torch.Tensor | None): 填充掩码 (key的padding), 形状 [B, S_k]。
                                                True为非padding, False为padding。
        返回:
            torch.Tensor: 注意力权重, 形状 [B, H, S_q, S_k]。
        """
        d_k = q_heads.size(-1)
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(d_k) # [B, H, S_q, S_k]

        if padding_mask is not None:
            # padding_mask: [B, S_k] -> attn_padding_mask: [B, 1, 1, S_k]
            attn_padding_mask = (padding_mask == 0).unsqueeze(1).unsqueeze(2) # True为padding位置
            scores = scores.masked_fill(attn_padding_mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights = self.dropout(attn_weights) # 可选的dropout
        return attn_weights


# --- 模块 5: 差分注意力核心逻辑 ---
class DifferentialAttentionCore(nn.Module):
    """
    计算差分注意力图的核心逻辑。
    输出: diff_attn_map = softmax(A1) - λ * softmax(A2)
    """
    def __init__(self):
        super().__init__()
        self.sdpa1 = ScaledDotProductAttention() # 实例化基础的点积注意力
        self.sdpa2 = ScaledDotProductAttention()

    def forward(self,
                q1_heads: torch.Tensor, k1_heads: torch.Tensor,
                q2_heads: torch.Tensor, k2_heads: torch.Tensor,
                learned_lambda: nn.Parameter,
                padding_mask: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数:
            q1_heads, k1_heads, q2_heads, k2_heads: 两组Q, K头, 形状 [B, H, S, D_h]。
            learned_lambda (nn.Parameter): 可学习的λ标量。
            padding_mask (torch.Tensor | None): 填充掩码 (key的padding), 形状 [B, S_k]。
        返回:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - diff_attn_map (torch.Tensor): 差分注意力图, 形状 [B, H, S_q, S_k]。
                - attn_weights1 (torch.Tensor): 第一个softmax注意力权重。
                - attn_weights2 (torch.Tensor): 第二个softmax注意力权重。
        """
        attn_weights1 = self.sdpa1(q1_heads, k1_heads, padding_mask)
        attn_weights2 = self.sdpa2(q2_heads, k2_heads, padding_mask)

        diff_attn_map = attn_weights1 - learned_lambda * attn_weights2
        return diff_attn_map, attn_weights1, attn_weights2


# --- 模块 6: 多头QKV投影 (针对差分注意力) ---
class DiffMultiHeadProjection(nn.Module):
    """
    将输入x投影到差分注意力所需的多头Q1, Q2, K1, K2, V。
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # WQ, WK 将 d_model 映射到 n_heads * d_head * 2 (因为有Q1,K1和Q2,K2两组)
        self.wq_linear = nn.Linear(d_model, d_model * 2)
        self.wk_linear = nn.Linear(d_model, d_model * 2)
        # WV 将 d_model 映射到 n_heads * d_head
        self.wv_linear = nn.Linear(d_model, d_model)

    def forward(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数:
            x_norm (torch.Tensor): 归一化后的输入, 形状 [B, S, D_model]。
        返回:
            q1_h, k1_h, q2_h, k2_h, v_h: 投影后的QKV头, 形状 [B, H, S, D_h]。
        """
        batch_size, seq_len, _ = x_norm.shape

        q_proj = self.wq_linear(x_norm)  # [B, S, D_model*2]
        k_proj = self.wk_linear(x_norm)  # [B, S, D_model*2]
        v_proj = self.wv_linear(x_norm)  # [B, S, D_model] (V不拆分)

        q1_h = q_proj[:, :, :self.d_model].view(batch_size, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        q2_h = q_proj[:, :, self.d_model:].view(batch_size, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k1_h = k_proj[:, :, :self.d_model].view(batch_size, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k2_h = k_proj[:, :, self.d_model:].view(batch_size, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v_h = v_proj.view(batch_size, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        return q1_h, k1_h, q2_h, k2_h, v_h


# --- 模块 7: (主) 带层级掩码的差分多头注意力 ---
class HierarchicalDifferentialAttention(nn.Module):
    """
    集成了QKV投影、RoPE应用、差分注意力核心、层级掩码应用、上下文计算、
    GroupNorm、λ_init缩放和输出投影。
    """
    def __init__(self, d_model: int, n_heads: int, rotary_emb_instance: RotaryEmbedding, # 接收RoPE实例
                 lambda_init_base: float = 0.8, lambda_init_scale: float = 0.6,
                 lambda_init_factor: float = -0.3):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads # 用于确认RoPE作用维度

        self.projection = DiffMultiHeadProjection(d_model, n_heads)
        self.rotary_emb = rotary_emb_instance # 存储RoPE实例
        if self.rotary_emb.dim != self.d_head:
            raise ValueError(f"RoPE维度({self.rotary_emb.dim})与注意力头维度({self.d_head})不匹配。")

        self.diff_core = DifferentialAttentionCore()
        self.out_linear = nn.Linear(d_model, d_model)
        self.group_norm = nn.GroupNorm(n_heads, d_model)

        self.lambda_init_base = lambda_init_base
        self.lambda_init_scale = lambda_init_scale
        self.lambda_init_factor = lambda_init_factor
        self.current_lambda_init_for_scaling = lambda_init_base

    def set_lambda_init_for_scaling(self, layer_idx: int):
        l_minus_1 = float(layer_idx)
        self.current_lambda_init_for_scaling = self.lambda_init_base - \
                                               self.lambda_init_scale * \
                                               math.exp(self.lambda_init_factor * l_minus_1)

    def forward(self, x_norm: torch.Tensor,
                learned_lambda: nn.Parameter,
                hierarchical_mask_C: torch.Tensor,
                padding_mask: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x_norm.shape

        # 1. QKV投影
        q1_h_no_rope, k1_h_no_rope, q2_h_no_rope, k2_h_no_rope, v_h = self.projection(x_norm)

        # 1.5 应用RoPE到Q和K的各个头上
        # RoPE的forward方法期望输入形状如 [B, H, S, D_h]
        q1_h = self.rotary_emb(q1_h_no_rope)
        k1_h = self.rotary_emb(k1_h_no_rope)
        q2_h = self.rotary_emb(q2_h_no_rope)
        k2_h = self.rotary_emb(k2_h_no_rope)
        # V (v_h) 通常不应用RoPE

        # 2. 计算差分注意力图
        diff_attn_map, attn_w1, attn_w2 = self.diff_core(
            q1_h, k1_h, q2_h, k2_h, learned_lambda, padding_mask
        )

        # 3. 应用层级掩码 C
        combined_attn_map = hierarchical_mask_C * diff_attn_map

        # 4. 计算上下文向量
        context_heads = torch.matmul(combined_attn_map, v_h)

        # 5. 合并多头
        context_merged = context_heads.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)

        # 6. 应用GroupNorm
        context_normed = self.group_norm(context_merged.transpose(1, 2)).transpose(1, 2)

        # 7. 应用 (1 - λ_init) 缩放
        scaled_context = context_normed * (1.0 - self.current_lambda_init_for_scaling)

        # 8. 输出投影
        output = self.out_linear(scaled_context)

        return output, (attn_w1, attn_w2)


# --- 模块 8: SwiGLU 前馈网络 ---
class SwiGLUFFN(nn.Module):
    """
    SwiGLU 前馈网络。 FFN(x) = (Swish(x W_g) * (x W_1)) W_2
    """
    def __init__(self, d_model: int, d_ff: int, dropout_rate_ffn: float = 0.1):
        super().__init__()
        self.w_g = nn.Linear(d_model, d_ff, bias=False) # 通常SwiGLU的门和主路径不加偏置
        self.w_1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout_ffn_internal = nn.Dropout(dropout_rate_ffn) # Dropout在最终输出前

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_val = F.silu(self.w_g(x))    # Swish(x W_g)
        hidden_val = self.w_1(x)          # x W_1
        # Dropout 在 w_2 之前
        return self.w_2(self.dropout_ffn_internal(gate_val * hidden_val))


# --- 模块 9: 带差分注意力的层级编码器层 ---
class HierarchicalDiffAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, rotary_emb_instance: RotaryEmbedding, # 接收RoPE实例
                 dropout_rate: float = 0.1,
                 layer_idx: int = 0,  # 当前层索引 (0-indexed)
                 d_k_neighbor: int | None = None, # 用于邻接亲和度计算
                 eps_norm: float = 1e-5 # RMSNorm的eps
                ):
        super().__init__()
        self.layer_idx = layer_idx # 保存层索引，用于亲和度计算

        # 实例化邻接亲和度计算器 (如果需要)
        if d_k_neighbor is not None:
            self.affinity_calculator = NeighborAffinityCalculator(d_model, d_k_neighbor)
        else: # 如果不提供d_k_neighbor，则不使用层级特性，C掩码将为全1
            self.affinity_calculator = None
            print(f"警告: Layer {layer_idx} 未提供 d_k_neighbor, 层级掩码C将为全1。")

        # 实例化带层级掩码的差分注意力模块
        self.diff_attn_hierarchical = HierarchicalDifferentialAttention(
            d_model, n_heads, rotary_emb_instance
        )
        # 为该层的差分注意力设置其 λ_init 值 (用于 (1-λ_init) 缩放)
        self.diff_attn_hierarchical.set_lambda_init_for_scaling(layer_idx)

        # 初始化该层可学习的 λ
        # λ_init = 0.8 - 0.6 * exp(-0.3 * l_minus_1)
        l_minus_1 = float(layer_idx)
        initial_lambda_val = self.diff_attn_hierarchical.lambda_init_base - \
                             self.diff_attn_hierarchical.lambda_init_scale * \
                             math.exp(self.diff_attn_hierarchical.lambda_init_factor * l_minus_1)
        self.learned_lambda = nn.Parameter(torch.tensor(initial_lambda_val, dtype=torch.float))

        # SwiGLU FFN
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout_rate_ffn=dropout_rate) # FFN内部dropout

        # 层归一化 (RMSNorm)
        self.norm1 = RMSNorm(d_model, eps=eps_norm) # 注意力子层前的LN
        self.norm2 = RMSNorm(d_model, eps=eps_norm) # FFN子层前的LN

        # 子层输出的Dropout (在残差连接和Add之前)
        self.dropout_sublayer_attn = nn.Dropout(dropout_rate)
        self.dropout_sublayer_ffn = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor,
                src_padding_mask: torch.Tensor,
                prev_affinity_scores_a: torch.Tensor | None
               ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        current_affinity_scores_a_out = None

        # 1. 计算邻接亲和度 'a' 和层级注意力掩码 'C'
        if self.affinity_calculator is not None:
            current_affinity_scores_a_out = self.affinity_calculator(
                x, # 使用上一层的输出x计算当前层的亲和度
                prev_affinity_scores_a,
                self.layer_idx # 使用当前层的索引
            )
            hierarchical_mask_C = build_hierarchical_attention_mask(
                current_affinity_scores_a_out, seq_len, x.device
            )
        else: # 如果没有亲和度计算器，C是全1的掩码，相当于不使用层级特性
            hierarchical_mask_C = torch.ones(batch_size, 1, seq_len, seq_len, device=x.device, dtype=x.dtype)


        # 2. 带层级掩码的差分多头注意力子层
        residual = x
        x_norm = self.norm1(x) # Pre-RMSNorm
        attn_output, attention_weights_tuple = self.diff_attn_hierarchical(
            x_norm,
            learned_lambda=self.learned_lambda,
            hierarchical_mask_C=hierarchical_mask_C,
            padding_mask=src_padding_mask
        )
        # Dropout 在注意力模块的输出上，残差连接之前
        x = residual + self.dropout_sublayer_attn(attn_output)

        # 3. SwiGLU FFN 子层
        residual_ffn = x
        x_norm_ffn = self.norm2(x) # Pre-RMSNorm
        ffn_output = self.ffn(x_norm_ffn)
        # Dropout 在FFN模块的输出上，残差连接之前
        x = residual_ffn + self.dropout_sublayer_ffn(ffn_output)

        return x, current_affinity_scores_a_out, attention_weights_tuple


# --- 模块 10 初步预测头 (PreliminaryPredictionHead) ---
class PreliminaryPredictionHead(nn.Module):
    """
    根据论文公式 (1) 进行初步的槽位(Slot)和意图(Intent)预测。

    它将每个时间步的编码器输出 h_j 与整个序列的池化表示 Pooled(h) 拼接起来，
    然后通过线性层进行预测。
    """
    def __init__(self, d_model: int, num_slot_labels: int, num_intent_labels: int):
        """
        初始化初步预测头。

        参数:
            d_model (int): 编码器输出的维度。
            num_slot_labels (int): 槽位标签的数量 (d_s)。如果为0，则不进行槽位预测。
            num_intent_labels (int): 意图标签的数量 (d_i)。如果为0，则不进行意图预测。
        """
        super().__init__()
        self.ds = num_slot_labels
        self.di = num_intent_labels

        # 用于预测 yS 和 yI 的线性层 - 公式 (1)
        # 输入维度是 2 * d_model (因为是 h_j || Pooling(h) )
        if self.di > 0: # 只有在需要意图预测时才定义意图预测线性层
            self.Wi_linear = nn.Linear(d_model * 2, self.di)
        if self.ds > 0: # 只有在需要槽位预测时才定义槽位预测线性层
            self.Ws_linear = nn.Linear(d_model * 2, self.ds)

    def _masked_average_pool(self, h: torch.Tensor, src_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        执行带掩码的平均池化操作，忽略padding部分。

        参数:
            h (torch.Tensor): 编码器的隐藏状态输出，形状 [batch_size, seq_len, d_model]。
            src_padding_mask (torch.Tensor): 源序列的padding掩码，形状 [batch_size, seq_len]。
                                             True表示非padding部分。

        返回:
            torch.Tensor: 池化后的句子表示，形状 [batch_size, d_model]。
        """
        # 将padding位置的h置为0，以便不影响求和
        masked_h = h * src_padding_mask.unsqueeze(-1).float()
        # 沿序列长度维度求和
        sum_h = torch.sum(masked_h, dim=1)                     # [batch_size, d_model]
        # 计算每个序列的实际长度 (非padding部分的数量)
        num_non_padding = src_padding_mask.sum(dim=1, keepdim=True).float() # [batch_size, 1]
        # 防止除以零 (如果整个序列都是padding，虽然不太可能在有效输入中出现)
        num_non_padding = torch.clamp(num_non_padding, min=1.0)
        return sum_h / num_non_padding

    def forward(self, h: torch.Tensor, src_padding_mask: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        前向传播。

        参数:
            h (torch.Tensor): 编码器的最终输出，形状 [batch_size, seq_len, d_model]。
            src_padding_mask (torch.Tensor): 源序列的padding掩码，形状 [batch_size, seq_len]。
                                             True表示非padding部分。

        返回:
            tuple[torch.Tensor | None, torch.Tensor | None]:
                - yI_prelim (torch.Tensor | None): 初步的意图预测，形状 [batch_size, seq_len, num_intent_labels] 或 None。
                - yS_prelim (torch.Tensor | None): 初步的槽位预测，形状 [batch_size, seq_len, num_slot_labels] 或 None。
        """
        yS_prelim, yI_prelim = None, None

        # 如果不需要进行任何预测，则直接返回
        if self.ds <= 0 and self.di <= 0:
            return yI_prelim, yS_prelim

        # 1. 计算句子的池化表示 Pooled(h)
        pooled_h_sentence = self._masked_average_pool(h, src_padding_mask) # [B, D]

        # todo 在这里进行pooling到底是好还是坏
        # todo 其次你必须把它从这里摘出来
        # 2. 准备拼接特征
        seq_len = h.size(1)
        # 将 pooled_h_sentence 扩展到与 h 的序列长度维度一致，以便拼接
        # pooled_h_expanded 形状: [batch_size, seq_len, d_model]
        pooled_h_expanded = pooled_h_sentence.unsqueeze(1).expand(-1, seq_len, -1)

        # 3. 拼接 h_j 和 Pooled(h)
        # combined_features 形状: [batch_size, seq_len, d_model * 2]
        combined_features = torch.cat((h, pooled_h_expanded), dim=-1)

        # 4. 通过线性层进行预测
        if self.ds > 0:
            yS_prelim = self.Ws_linear(combined_features)  # [B, S, ds]
        if self.di > 0:
            yI_prelim = self.Wi_linear(combined_features)  # [B, S, di]

        return yI_prelim, yS_prelim


# --- 主模型: 层级差分注意力编码器 ---
class HierarchicalDiffEncoderWithRoPE(nn.Module):
    """主编码器模型，集成了层级差分注意力机制，并使用RoPE进行位置编码。"""
    def __init__(self, num_encoder_layers: int, d_model: int, n_heads: int, d_ff: int,
                 input_vocab_size: int, max_seq_len: int, # max_seq_len 用于RoPE
                 num_slot_labels: int, num_intent_labels: int,
                 dropout_rate: float = 0.1, padding_idx: int = 0,
                 d_k_neighbor: int | None = None,
                 rope_theta: float = 10000.0 # RoPE的theta参数
                ):
        super().__init__()
        self.padding_idx = padding_idx
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_k_neighbor = d_k_neighbor

        self.token_embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=self.padding_idx)
        # 按照方案A，在embedding后应用dropout
        self.embedding_dropout = nn.Dropout(dropout_rate)
        # 移除 self.pos_encoder

        # 实例化RoPE模块，它将被所有层共享
        self.rope_emb = RotaryEmbedding(dim=self.d_head, max_seq_len=max_seq_len, theta=rope_theta)

        self.layers = nn.ModuleList([
            HierarchicalDiffAttentionEncoderLayer(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                rotary_emb_instance=self.rope_emb, # 将RoPE实例传递给每一层
                dropout_rate=dropout_rate, layer_idx=i, d_k_neighbor=d_k_neighbor
            ) for i in range(num_encoder_layers)
        ])

        self.prelim_predictor = None
        if num_slot_labels > 0 or num_intent_labels > 0:
            self.prelim_predictor = PreliminaryPredictionHead(d_model, num_slot_labels, num_intent_labels)

    def forward(self, src_tokens: torch.Tensor) -> dict:
        src_padding_mask = (src_tokens != self.padding_idx)
        x = self.token_embedding(src_tokens) * math.sqrt(self.d_model)
        x = self.embedding_dropout(x) # 应用词嵌入后的dropout

        current_affinity_scores_a_inter_layer = None
        all_layer_attention_weight_tuples = []

        for layer in self.layers:
            x, affinity_out, attention_weights_tuple = layer(
                x, src_padding_mask, current_affinity_scores_a_inter_layer
            )
            if self.d_k_neighbor is not None: # 只有在启用了层级特性时才更新
                current_affinity_scores_a_inter_layer = affinity_out
            all_layer_attention_weight_tuples.append(attention_weights_tuple)

        h = x
        yI_prelim, yS_prelim = None, None
        if self.prelim_predictor is not None:
            yI_prelim, yS_prelim = self.prelim_predictor(h, src_padding_mask)

        final_affinity_to_return = current_affinity_scores_a_inter_layer if self.d_k_neighbor is not None else None

        return {
            "encoder_output": h,
            "prelim_intent_predictions": yI_prelim,
            "prelim_slot_predictions": yS_prelim,
            "final_affinity_scores_a": final_affinity_to_return,
            "all_layer_attention_weights": all_layer_attention_weight_tuples,
            "source_padding_mask": src_padding_mask
        }

# --- 示例用法 ---
if __name__ == '__main__':
    vocab_size = 1000
    d_model = 128
    n_heads = 4
    # 对于SwiGLU, d_ff 通常是 d_model * (8/3) 或 d_model * 4。
    # 但为了保持与之前示例的参数量相似性（如果PositionalEncoding很大），我们可能需要调整。
    # 论文中Diff Transformer FFN size 是 8/3 * d_model * 2 (因为SwiGLU有两个线性层到d_ff)
    # 这里简化 d_ff = d_model * 2 (指SwiGLU的中间维度)
    d_ff = d_model * 2 # SwiGLU 通常用 d_ff = d_model * 8/3 * 2，这里简化
    num_enc_layers = 2
    max_seq_len_for_rope = 60 # 用于RoPE
    dropout = 0.1
    pad_idx = 0
    num_slots = 5
    num_intents = 2
    dk_neighbor_val = int(math.sqrt(d_model))

    print("--- 测试带RoPE和层级特性的层级差分注意力编码器 ---")
    hier_diff_encoder_with_rope = HierarchicalDiffEncoderWithRoPE(
        num_encoder_layers=num_enc_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        input_vocab_size=vocab_size,
        max_seq_len=max_seq_len_for_rope, # 传递给RoPE
        num_slot_labels=num_slots,
        num_intent_labels=num_intents,
        dropout_rate=dropout,
        padding_idx=pad_idx,
        d_k_neighbor=dk_neighbor_val
    )

    batch = 2; seq_len_val = 10 # 实际序列长度可以小于max_seq_len_for_rope
    dummy_tokens = torch.randint(1, vocab_size, (batch, seq_len_val))
    dummy_tokens[0, -3:] = pad_idx

    hier_diff_encoder_with_rope.eval()
    with torch.no_grad():
        outputs_rope = hier_diff_encoder_with_rope(dummy_tokens)

    print(f"带RoPE的编码器输出形状: {outputs_rope['encoder_output'].shape}")
    if outputs_rope["prelim_slot_predictions"] is not None:
        print(f"槽位预测形状: {outputs_rope['prelim_slot_predictions'].shape}")
    if outputs_rope["final_affinity_scores_a"] is not None:
        print(f"最终亲和度分数形状: {outputs_rope['final_affinity_scores_a'].shape}")

    print(f"注意力权重元组列表长度: {len(outputs_rope['all_layer_attention_weights'])}")
    if outputs_rope['all_layer_attention_weights']:
        attn_tuple_l0_rope = outputs_rope['all_layer_attention_weights'][0]
        print(f"  第0层注意力权重元组长度: {len(attn_tuple_l0_rope)}")
        print(f"    第0层attn_weights1形状: {attn_tuple_l0_rope[0].shape}")
        print(f"    第0层attn_weights2形状: {attn_tuple_l0_rope[1].shape}")
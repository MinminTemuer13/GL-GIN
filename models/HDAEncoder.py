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


# --- 模块 1: 位置编码 (PositionalEncoding) ---
class PositionalEncoding(nn.Module):
    """标准的Transformer位置编码模块。

    通过在词嵌入中加入位置信息，使得模型能够区分序列中不同位置的词。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        初始化位置编码模块。

        参数:
            d_model (int): 模型的维度 (词嵌入的维度)。
            dropout (float): Dropout的比率。
            max_len (int): 支持的最大序列长度。
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够大的位置编码矩阵(pe)
        pe = torch.zeros(max_len, d_model)
        # 生成位置信息 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算除数项，用于sin和cos函数
        # div_term 的形状是 [d_model/2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数维度使用sin函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度使用cos函数
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加一个批次维度，使其形状为 [1, max_len, d_model] 以便广播
        pe = pe.unsqueeze(0)
        # 将pe注册为buffer，这样它不会被视为模型参数，但会随模型移动(例如.to(device))
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]。

        返回:
            torch.Tensor: 加入了位置编码的输出张量，形状与输入相同。
        """
        # 将x与对应序列长度的位置编码相加
        # self.pe的形状是 [1, max_len, d_model]，通过切片取前seq_len个位置
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


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


# --- 模块 4: 带层级掩码的差分多头注意力 ---
class DifferentialMultiHeadAttentionWithHierarchicalMask(nn.Module):
    """
    差分多头注意力机制，并结合了外部传入的层级掩码C。
    """
    def __init__(self, d_model: int, n_heads: int, dropout_rate_attention: float = 0.0, # DIFF论文未明确注意力dropout
                 lambda_init_base: float = 0.8, lambda_init_scale: float = 0.6,
                 lambda_init_factor: float = -0.3):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale_factor = math.sqrt(self.d_head)

        self.wq_linear = nn.Linear(d_model, d_model * 2) # 投影到 (Q1, Q2)
        self.wk_linear = nn.Linear(d_model, d_model * 2) # 投影到 (K1, K2)
        self.wv_linear = nn.Linear(d_model, d_model)     # 投影到 V (共享)
        self.out_linear = nn.Linear(d_model, d_model)

        # GroupNorm，每个头独立归一化
        self.group_norm = nn.GroupNorm(self.n_heads, self.d_model) # 对拼接后的d_model (C) 进行操作

        # Dropout应用在 (DiffAttnWeights * C) @ V 之后，由外部EncoderLayer的dropout_sublayer处理

        # λ_init 相关参数存储，用于计算固定的 (1 - λ_init) 缩放因子
        self.lambda_init_base = lambda_init_base
        self.lambda_init_scale = lambda_init_scale
        self.lambda_init_factor = lambda_init_factor
        self.current_lambda_init = lambda_init_base # 会被 set_lambda_init 更新

    def set_lambda_init_for_scaling(self, layer_idx: int):
        """
        根据层索引计算并存储当前层的 λ_init 值，用于 (1-λ_init) 缩放。
        此方法应在模块实例化后，但在第一次forward之前被外部调用。
        """
        # 论文中 l ∈ [1, L], 若我们的 layer_idx 是 0-indexed, 则 l-1 对应 layer_idx
        l_minus_1 = float(layer_idx)
        self.current_lambda_init = self.lambda_init_base - self.lambda_init_scale * math.exp(self.lambda_init_factor * l_minus_1)

    def forward(self, x_norm: torch.Tensor, # 输入已经是归一化后的
                learned_lambda: nn.Parameter,
                hierarchical_mask_C: torch.Tensor, # 外部传入的层级掩码C
                padding_mask: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x_norm.shape

        q_proj = self.wq_linear(x_norm)
        k_proj = self.wk_linear(x_norm)
        v_proj = self.wv_linear(x_norm)

        q1 = q_proj[:, :, :self.d_model].view(batch_size, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        q2 = q_proj[:, :, self.d_model:].view(batch_size, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k1 = k_proj[:, :, :self.d_model].view(batch_size, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k2 = k_proj[:, :, self.d_model:].view(batch_size, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = v_proj.view(batch_size, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) / self.scale_factor
        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) / self.scale_factor

        if padding_mask is not None:
            attn_padding_mask = (padding_mask == 0).unsqueeze(1).unsqueeze(2)
            scores1 = scores1.masked_fill(attn_padding_mask, -1e9)
            scores2 = scores2.masked_fill(attn_padding_mask, -1e9)

        attn_weights1 = F.softmax(scores1, dim=-1)
        attn_weights2 = F.softmax(scores2, dim=-1)

        # 计算差分注意力图
        # diff_attn_map 形状: [B, H, S, S]
        diff_attn_map = attn_weights1 - learned_lambda * attn_weights2

        # 方案A: 将差分注意力图与层级掩码C结合
        # hierarchical_mask_C: [B, 1, S, S]，会自动广播
        # 注意：diff_attn_map可能包含负值。C通常是[0,1]之间的值。
        # 乘积结果的解释性：C可以看作是路径的“可信度”或“连通性”，
        # 它调节了差分注意力信号的强度。
        combined_attn_map = hierarchical_mask_C * diff_attn_map

        # 与 V 相乘得到上下文 (此时还未应用GroupNorm和λ_init缩放)
        context_heads = torch.matmul(combined_attn_map, v) # [B, H, S, D_h]

        # 合并多头
        context_merged = context_heads.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model) # [B, S, D]

        # 应用 GroupNorm
        # 输入需要是 (N, C, L), 所以 [B, D, S]
        context_normed = self.group_norm(context_merged.transpose(1, 2)).transpose(1, 2) # [B, S, D]

        # 应用 (1 - λ_init) 缩放因子
        scaled_context = context_normed * (1.0 - self.current_lambda_init)

        output = self.out_linear(scaled_context)

        return output, (attn_weights1, attn_weights2) # 返回原始的两个softmax权重图用于分析


# --- 模块 5: SwiGLU 前馈网络 ---
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


# --- 模块 6: 带差分注意力的层级编码器层 ---
class HierarchicalDiffAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float = 0.1,
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
        self.diff_attn_hierarchical = DifferentialMultiHeadAttentionWithHierarchicalMask(
            d_model, n_heads, dropout_rate_attention=0.0 # DIFF论文未明确注意力内部dropout
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


# --- 模块 7: 初步预测头 (PreliminaryPredictionHead) ---
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


# --- 模块 8: 层级差分注意力编码器 (主模型) ---
class HierarchicalDiffEncoder(nn.Module):
    def __init__(self, num_encoder_layers: int, d_model: int, n_heads: int, d_ff: int,
                 input_vocab_size: int, max_len: int,
                 num_slot_labels: int, num_intent_labels: int,
                 dropout_rate: float = 0.1, padding_idx: int = 0,
                 d_k_neighbor: int | None = None # 如果为None，则不启用层级特性
                ):
        super().__init__()
        self.padding_idx = padding_idx
        self.d_model = d_model
        self.d_k_neighbor = d_k_neighbor

        self.token_embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=self.padding_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate, max_len)

        self.layers = nn.ModuleList([
            HierarchicalDiffAttentionEncoderLayer(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                dropout_rate=dropout_rate,
                layer_idx=i,
                d_k_neighbor=d_k_neighbor
            )
            for i in range(num_encoder_layers)
        ])

        self.prelim_predictor = None
        if num_slot_labels > 0 or num_intent_labels > 0:
            self.prelim_predictor = PreliminaryPredictionHead(d_model, num_slot_labels, num_intent_labels)

    def forward(self, src_tokens: torch.Tensor) -> dict:
        src_padding_mask = (src_tokens != self.padding_idx)
        x = self.token_embedding(src_tokens) * math.sqrt(self.d_model)
        x = self.pos_encoder(x) # Dropout在PositionalEncoding内部

        current_affinity_scores_a_inter_layer = None
        all_layer_attention_weight_tuples = []

        for layer in self.layers:
            x, affinity_out, attention_weights_tuple = layer(
                x,
                src_padding_mask=src_padding_mask,
                prev_affinity_scores_a=current_affinity_scores_a_inter_layer
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
    d_ff = d_model * 2 # SwiGLU 通常用 d_ff = d_model * 8/3 * 2，这里简化
    num_enc_layers = 2
    max_seq_len = 60
    dropout = 0.1
    pad_idx = 0
    num_slots = 5
    num_intents = 2
    dk_neighbor_val = int(math.sqrt(d_model))

    print("--- 测试带层级特性的层级差分注意力编码器 ---")
    hier_diff_encoder = HierarchicalDiffEncoder(
        num_encoder_layers=num_enc_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        input_vocab_size=vocab_size, max_len=max_seq_len,
        num_slot_labels=num_slots, num_intent_labels=num_intents,
        dropout_rate=dropout, padding_idx=pad_idx,
        d_k_neighbor=dk_neighbor_val # 启用层级特性
    )

    batch = 2
    seq_len_val = 10
    dummy_tokens = torch.randint(1, vocab_size, (batch, seq_len_val))
    dummy_tokens[0, -3:] = pad_idx

    hier_diff_encoder.eval()
    with torch.no_grad():
        outputs = hier_diff_encoder(dummy_tokens)

    print(f"编码器输出形状: {outputs['encoder_output'].shape}")
    if outputs["prelim_slot_predictions"] is not None:
        print(f"槽位预测形状: {outputs['prelim_slot_predictions'].shape}")
    if outputs["final_affinity_scores_a"] is not None:
        print(f"最终亲和度分数形状: {outputs['final_affinity_scores_a'].shape}")
    else:
        print("最终亲和度分数为 None (可能未启用层级特性或序列过短)")

    print(f"注意力权重元组列表长度: {len(outputs['all_layer_attention_weights'])}")
    if outputs['all_layer_attention_weights']:
        attn_tuple_l0 = outputs['all_layer_attention_weights'][0]
        print(f"  第0层注意力权重元组长度: {len(attn_tuple_l0)}")
        print(f"    第0层attn_weights1形状: {attn_tuple_l0[0].shape}")
        print(f"    第0层attn_weights2形状: {attn_tuple_l0[1].shape}")

    print("\n--- 测试不带层级特性的层级差分注意力编码器 (C将为全1) ---")
    # d_k_neighbor=None 会使得C掩码为全1，相当于标准的差分注意力
    plain_diff_encoder = HierarchicalDiffEncoder(
        num_encoder_layers=num_enc_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        input_vocab_size=vocab_size, max_len=max_seq_len,
        num_slot_labels=num_slots, num_intent_labels=num_intents,
        dropout_rate=dropout, padding_idx=pad_idx,
        d_k_neighbor=None # 不启用层级特性
    )
    plain_diff_encoder.eval()
    with torch.no_grad():
        outputs_plain = plain_diff_encoder(dummy_tokens)
    print(f"编码器输出形状: {outputs_plain['encoder_output'].shape}")
    if outputs_plain["final_affinity_scores_a"] is not None:
        print(f"最终亲和度分数形状: {outputs_plain['final_affinity_scores_a'].shape}")
    else:
        print("最终亲和度分数为 None (因为d_k_neighbor=None)")
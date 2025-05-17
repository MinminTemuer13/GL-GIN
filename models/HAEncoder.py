import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """标准的Transformer位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class HierarchicalAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k_neighbor, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads # 标准多头注意力中每个头的维度
        self.d_k_neighbor = d_k_neighbor # 论文中 s_i,i+1 公式中的 d_s，邻接注意力分数的缩放因子

        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        # 用于计算邻接注意力分数 (s_i,i+1) - 公式(3)
        # 论文提到 W_Q 和 W_K 是可学习的矩阵。这里我们假设它们将 d_model 映射到 d_model
        # 并且这个计算是独立于标准多头注意力的 Q, K 的。
        self.wq_neighbor = nn.Linear(d_model, d_model)
        self.wk_neighbor = nn.Linear(d_model, d_model)

        # 标准多头注意力组件
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout), # 通常在FFN的隐藏层后加dropout
            nn.Linear(d_ff, d_model)
        )

        # 层归一化和Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.scale_factor = math.sqrt(self.d_head)

    def calculate_neighboring_affinity(self, x_input_for_affinity, prev_affinity_scores_a=None, layer_idx=0):
        """
        计算当前层的邻接亲和度分数 a_i,i+1
        x_input_for_affinity: 当前层用于计算亲和度的输入 [batch_size, seq_len, d_model]
        prev_affinity_scores_a: 上一层计算得到的 a 分数 [batch_size, seq_len-1]
        layer_idx: 当前层的索引 (0-indexed)
        """
        batch_size, seq_len, d_model_dim = x_input_for_affinity.shape
        if seq_len <= 1: # 序列长度不足以计算邻接分数
            return None

        # --- 公式 (3): 计算邻接注意力分数 s_i,i+1 ---
        # x_i W^Q
        q_for_s = self.wq_neighbor(x_input_for_affinity[:, :-1, :])  # [batch, seq_len-1, d_model]
        # x_{i+1} W^K
        k_for_s = self.wk_neighbor(x_input_for_affinity[:, 1:, :])   # [batch, seq_len-1, d_model]

        # (x_i W^Q) (x_{i+1} W^K)^T
        # 这里我们假设点积是在整个 d_model 维度上，然后除以 d_s (self.d_k_neighbor)
        # 论文中没有明确说明这个点积是否也按多头方式进行，通常这种辅助分数会简化处理。
        s_forward = torch.sum(q_for_s * k_for_s, dim=-1) / self.d_k_neighbor # [batch, seq_len-1]

        # 计算 s_{i+1,i} (反向)
        q_for_s_rev = self.wq_neighbor(x_input_for_affinity[:, 1:, :])
        k_for_s_rev = self.wk_neighbor(x_input_for_affinity[:, :-1, :])
        s_backward = torch.sum(q_for_s_rev * k_for_s_rev, dim=-1) / self.d_k_neighbor # [batch, seq_len-1]

        # --- 公式 (4): 计算邻接亲和度分数 â_i,i+1 ---
        # 论文中是 (softmax(s_i,i+1) + softmax(s_i+1,i)) / 2
        # Softmax通常作用于一个分布。对于s_i,i+1，它代表了x_i想与x_i+1合并的趋势。
        # 如果一个token有多个可能的合并对象（比如在图或更复杂的结构中），softmax才有意义。
        # 对于简单的序列邻接，softmax(s_i,i+1) 如果只对单个值s_i,i+1操作，会恒等于1（如果s_i,i+1是标量）。
        # 因此，这里更合理的解释是：
        # 1. s_i,i+1 本身就是某种“分数”，不需要再softmax。
        # 2. 或者，softmax是针对一个token的所有潜在“邻居”的s分数（但在序列中只有1或2个邻居）。
        # 3. 考虑到论文引用了 (Wang, Lee, and Chen 2019) Tree Transformer，其中邻接分数用于树结构。
        # 为了代码的可运行性和简化，这里假设s_forward和s_backward已经是代表趋势的“分数”，
        # 我们用sigmoid将其映射到(0,1)区间，这与softmax的目标（产生类似概率的值）相近，
        # 并且对于只有两个方向的平均是合理的。
        # 如果需要严格的softmax，需要明确softmax的作用域。
        # **一个更忠于论文的解释可能是，softmax(s_i,i+1)是相对于其他可能的“分割点”或“合并强度”**
        # **但这里没有明确给出。我们采用sigmoid作为一种归一化手段。**
        affinity_hat_forward = torch.sigmoid(s_forward)
        affinity_hat_backward = torch.sigmoid(s_backward)
        current_layer_affinity_hat = (affinity_hat_forward + affinity_hat_backward) / 2  # â_i,i+1, [batch, seq_len-1]

        # --- 公式 (5): 层级更新亲和度分数 a^l_i,i+1 ---
        if layer_idx == 0: # l=0 的情况
            updated_affinity_scores_a = current_layer_affinity_hat
        else: # l>=1 的情况
            assert prev_affinity_scores_a is not None, "prev_affinity_scores_a 不能为空，当 layer_idx > 0"
            # a^{l-1}_{i,i+1} + (1 - a^{l-1}_{i,i+1}) * â^l_{i,i+1}
            updated_affinity_scores_a = prev_affinity_scores_a + \
                                     (1 - prev_affinity_scores_a) * current_layer_affinity_hat
        return updated_affinity_scores_a

    def build_attention_mask_C(self, affinity_scores_a, seq_len, device):
        """
        构建注意力掩码 C - 公式 (6)
        affinity_scores_a: 当前层计算得到的 a 分数 [batch_size, seq_len-1]
        seq_len: 序列长度
        device: 计算设备
        """
        if affinity_scores_a is None: # 如果序列长度 <= 1
            # 返回一个允许所有token互相注意的掩码 (虽然单token序列自注意力意义不大)
            return torch.ones(1, 1, seq_len, seq_len, device=device, dtype=torch.float)

        batch_size = affinity_scores_a.shape[0]
        # 初始化 C_i,j = 1 (当 i=j 时)
        C = torch.eye(seq_len, device=device, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1, 1) # [batch, seq_len, seq_len]

        # 计算 C_i,j 当 i < j 时: Π_{k=i}^{j-1} a_{k,k+1}
        # affinity_scores_a[:, k] 对应 a_{k,k+1}
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                # 路径上的亲和度 a_{i,i+1}, a_{i+1,i+2}, ..., a_{j-1,j}
                # 对应 affinity_scores_a 的索引是 i, i+1, ..., j-1
                if j - 1 >= i: # 确保路径存在
                    path_affinities = affinity_scores_a[:, i:j] # 切片得到 [batch_size, j-i]
                    C[:, i, j] = torch.prod(path_affinities, dim=1)
                    C[:, j, i] = C[:, i, j] # 对称

        return C.unsqueeze(1) # 增加head维度: [batch, 1, seq_len, seq_len]

    def forward(self, x, src_padding_mask, prev_affinity_scores_a, layer_idx):
        """
        x: 输入张量 [batch_size, seq_len, d_model]
        src_padding_mask: 源序列的padding掩码 [batch_size, seq_len], True表示非padding, False表示padding
        prev_affinity_scores_a: 上一层计算的 'a' 分数 [batch_size, seq_len-1]
        layer_idx: 当前层索引
        """
        batch_size, seq_len, _ = x.shape

        # 1. 计算当前层的亲和度分数 'a' 和注意力掩码 'C'
        # 输入给亲和度计算的是上一层的输出 x
        current_affinity_scores_a = self.calculate_neighboring_affinity(x, prev_affinity_scores_a, layer_idx)
        hierarchical_attention_mask_C = self.build_attention_mask_C(current_affinity_scores_a, seq_len, x.device)
        # hierarchical_attention_mask_C 形状: [batch, 1, seq_len, seq_len]

        # 2. 标准多头注意力计算
        # 残差连接的输入
        residual = x

        # 层归一化 (Pre-LN 变体，更稳定)
        x_norm = self.norm1(x)

        q = self.q_linear(x_norm)  # [batch, seq_len, d_model]
        k = self.k_linear(x_norm)
        v = self.v_linear(x_norm)

        # 拆分成多头
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3) # [batch, n_heads, seq_len, d_head]
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # 计算注意力分数 (QK^T / sqrt(d_head))
        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale_factor # [batch, n_heads, seq_len, seq_len]

        # 应用 padding 掩码
        # src_padding_mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        if src_padding_mask is not None:
            attention_padding_mask = src_padding_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, S]
            scores = scores.masked_fill(attention_padding_mask == 0, -1e9) # padding位置设为极小值

        # --- 公式 (2): H = (C ⊙ softmax(QK^T/√d_h)) V ---
        # 先计算 softmax
        attention_weights_raw = F.softmax(scores, dim=-1) # [batch, n_heads, seq_len, seq_len]

        # 应用层次注意力掩码 C (Hadamard product)
        # hierarchical_attention_mask_C: [batch, 1, seq_len, seq_len], 会自动广播到 n_heads
        attention_weights_hierarchical = hierarchical_attention_mask_C * attention_weights_raw

        # 注意：论文没有明确在乘以C后是否重新归一化。
        # 如果C使得某些行的和不再为1，而后续又没有归一化，可能会导致数值问题或信息损失。
        # 一种可能的处理是确保C的值在[0,1]且重新归一化，但我们严格按公式(2)的形态。
        # 如果需要，可以考虑在这里对 attention_weights_hierarchical 的最后一维进行重新归一化。
        # renorm_attention = attention_weights_hierarchical / (torch.sum(attention_weights_hierarchical, dim=-1, keepdim=True) + 1e-9)
        # context = torch.matmul(self.dropout_attention(renorm_attention), v)

        context = torch.matmul(self.dropout_attention(attention_weights_hierarchical), v) # [batch, n_heads, seq_len, d_head]

        # 合并多头
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model) # [batch, seq_len, d_model]
        attention_out = self.out_linear(context)

        # Add & Norm (第一个残差连接)
        x = residual + self.dropout_attention(attention_out) # 使用 self.dropout_attention 而非独立的dropout

        # 3. FFN 部分
        residual_ffn = x
        x_norm_ffn = self.norm2(x) # Pre-LN for FFN
        ffn_out = self.ffn(x_norm_ffn)

        # Add & Norm (第二个残差连接)
        x = residual_ffn + self.dropout_ffn(ffn_out)

        return x, current_affinity_scores_a, attention_weights_hierarchical

class HAEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, d_k_neighbor, d_ff,
                 input_vocab_size, max_len, num_slot_labels, num_intent_labels, dropout=0.1, padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=self.padding_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([
            HierarchicalAttentionEncoderLayer(d_model, n_heads, d_k_neighbor, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.d_model = d_model # 用于初步预测

        # --- 初步预测层 yS 和 yI - 公式 (1) ---
        # h_j || Pooling(h)
        # Pooling(h) 是 average pooling
        # WS ∈ R^{d_s × 2d_model}, WI ∈ R^{d_i × 2d_model}
        self.ds = num_slot_labels
        self.di = num_intent_labels
        if self.ds > 0 : # 只有在需要槽位预测时才定义
            self.Ws_linear = nn.Linear(d_model * 2, self.ds)
        if self.di > 0: # 只有在需要意图预测时才定义
            self.Wi_linear = nn.Linear(d_model * 2, self.di)


    def forward(self, src_tokens):
        """
        src_tokens: 输入的token IDs [batch_size, seq_len]
        """
        # 1. 创建 padding 掩码
        # True表示是padding，False表示不是padding。注意力中通常反过来用。
        # 我们这里创建的 src_padding_mask 是 True 表示非padding, False 表示 padding
        src_padding_mask = (src_tokens != self.padding_idx) # [batch, seq_len]

        # 2. Token Embedding 和 Positional Encoding
        x = self.token_embedding(src_tokens) * math.sqrt(self.d_model) # 乘以sqrt(d_model)是Transformer的常见做法
        x = self.pos_encoder(x) # 包含了dropout

        current_affinity_scores_a = None # 初始化，用于第一层
        all_layer_attention_weights = [] # 用于分析

        # 3. 通过 Encoder Layers
        for i, layer in enumerate(self.layers):
            x, current_affinity_scores_a, attention_weights = layer(x, src_padding_mask, current_affinity_scores_a, layer_idx=i)
            all_layer_attention_weights.append(attention_weights)

        # 最终的编码器输出 h
        h = x # [batch_size, seq_len, d_model]

        # --- 4. 初步预测 yS 和 yI - 公式 (1) ---
        yS_prelim = None
        yI_prelim = None

        if self.ds > 0 or self.di > 0:  # 只有当需要预测槽位或意图时才执行
            # Pooling(h) - Average pooling over sequence dimension
            # 要注意padding的影响，只对非padding部分进行平均
            # h: [batch, seq_len, d_model], src_padding_mask: [batch, seq_len]
            masked_h = h * src_padding_mask.unsqueeze(-1).float() # 将padding位置的h置为0
            sum_h = torch.sum(masked_h, dim=1) # [batch, d_model]
            num_non_padding = src_padding_mask.sum(dim=1, keepdim=True).float() # [batch, 1]
            num_non_padding = torch.clamp(num_non_padding, min=1.0) # 防止除以0
            pooled_h_sentence = sum_h / num_non_padding  # [batch, d_model]

            # 扩展 pooled_h_sentence 以便与 h 拼接
            # h: [batch, seq_len, d_model]
            # 组合 h_j 和 Pooled_h_sentence
            # h_j: [batch, d_model] (在循环中取)
            # pooled_h_sentence: [batch, d_model]
            seq_len = h.size(1)
            pooled_h_expanded = pooled_h_sentence.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, d_model]

            combined_features = torch.cat((h, pooled_h_expanded), dim=-1)  # [batch, seq_len, d_model * 2]

            if self.ds > 0:
                yS_prelim = self.Ws_linear(combined_features)  # [batch, seq_len, ds]
            if self.di > 0:
                yI_prelim = self.Wi_linear(combined_features)  # [batch, seq_len, di]

        return {
            "encoder_output": h, # [batch_size, seq_len, d_model]
            "prelim_slot_predictions": yS_prelim, # [batch_size, seq_len, num_slot_labels] or None
            "prelim_intent_predictions": yI_prelim, # [batch_size, seq_len, num_intent_labels] or None
            "final_affinity_scores_a": current_affinity_scores_a, # [batch_size, seq_len-1] or None
            "all_layer_attention_weights": all_layer_attention_weights # list of [batch, n_heads, seq_len, seq_len]
        }

# --- 示例用法 ---
if __name__ == '__main__':
    # 定义超参数 (与论文表1后的描述尽量一致或合理假设)
    vocab_size = 10000        # 假设的词汇表大小
    d_model = 128 + 256             # Transformer 输入输出维度 (论文Table1后 Ne=4, d_model=128)
    n_heads = 8               # 注意力头数 (论文Table1后 num_attention_heads=8)
    d_k_neighbor = math.sqrt(d_model) # 公式(3)中的 d_s, 论文未明确, 合理假设为sqrt(d_model)或d_model/n_heads
                                     # 或者一个固定的超参数, 例如论文引用[Wang et al. 2019]可能提到
                                     # 这里用 sqrt(d_model) 作为示例
    d_ff = d_model * 4        # FFN中间层维度, Transformer常见设置为4*d_model (论文未明确, 这是常见做法)
    num_layers = 4            # Encoder层数 (论文Table1后 Ne=4)
    max_seq_len = 64          # 假设的最大序列长度
    dropout_rate = 0.1        # Dropout比例 (论文Table1后 dropout_ratio=0.1)
    padding_idx = 0           # Padding token的ID

    num_slot_labels_example = 50 # 假设的槽位标签数量
    num_intent_labels_example = 10 # 假设的意图标签数量


    # 实例化Encoder
    encoder = HAEncoder(
        num_layers=num_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_k_neighbor=d_k_neighbor,
        d_ff=d_ff,
        input_vocab_size=vocab_size,
        max_len=max_seq_len,
        num_slot_labels=num_slot_labels_example,
        num_intent_labels=num_intent_labels_example,
        dropout=dropout_rate,
        padding_idx=padding_idx
    )

    # 创建模拟输入数据
    batch_size = 4
    seq_length = 30
    dummy_src_tokens = torch.randint(1, vocab_size, (batch_size, seq_length)) # 从1开始，0是padding
    # 随机加入一些padding
    for i in range(batch_size):
        pad_len = torch.randint(0, seq_length // 2, (1,)).item()
        if pad_len > 0:
            dummy_src_tokens[i, -pad_len:] = padding_idx

    print("模拟输入 (src_tokens):")
    print(dummy_src_tokens)
    print("形状:", dummy_src_tokens.shape)

    # 前向传播
    encoder_outputs = encoder(dummy_src_tokens)

    print("\nEncoder 输出 h 形状:", encoder_outputs["encoder_output"].shape)
    if encoder_outputs["prelim_slot_predictions"] is not None:
        print("初步槽位预测 yS 形状:", encoder_outputs["prelim_slot_predictions"].shape)
    if encoder_outputs["prelim_intent_predictions"] is not None:
        print("初步意图预测 yI 形状:", encoder_outputs["prelim_intent_predictions"].shape)
    if encoder_outputs["final_affinity_scores_a"] is not None:
        print("最后一层邻接亲和度 a 形状:", encoder_outputs["final_affinity_scores_a"].shape)
    print("每层注意力权重数量:", len(encoder_outputs["all_layer_attention_weights"]))
    if encoder_outputs["all_layer_attention_weights"]:
        print("第一层注意力权重形状:", encoder_outputs["all_layer_attention_weights"][0].shape)
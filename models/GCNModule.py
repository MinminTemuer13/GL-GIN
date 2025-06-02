import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedAffinityGCNLayer(nn.Module):
    def __init__(
            self,
            d_intent,
            d_slot,
            epsilon=1e-12,
            activation_fn_str="relu",
            affinity_power=1.0
    ):
        """
        增强的亲和力GCN层。
        参数:
            d_intent (int): 意图特征的维度。
            d_slot (int): 槽位特征的维度。
            epsilon (float): 一个小常数，用于防止除以零。
            activation_fn_str (str): 激活函数的名称 ("relu", "gelu", "tanh")。
            affinity_power (float): 用于增强亲和力矩阵对比度的幂次。
                                    大于1增强强连接，等于1不改变。
        """
        super().__init__()
        self.d_intent = d_intent
        self.d_slot = d_slot
        self.epsilon = epsilon

        if activation_fn_str == "relu":
            self.activation_fn = F.relu
        elif activation_fn_str == "gelu":
            self.activation_fn = F.gelu
        elif activation_fn_str == "tanh":
            self.activation_fn = torch.tanh
        else:
            raise ValueError(f"不支持的激活函数: {activation_fn_str}")

        self.affinity_power = affinity_power # 用于增强对比度

        # 可学习的权重矩阵，用于特征变换
        self.W_i2i = nn.Linear(d_intent, d_intent, bias=False) # 意图到意图
        self.W_s2i = nn.Linear(d_slot, d_intent, bias=False)   # 槽位到意图
        self.W_s2s = nn.Linear(d_slot, d_slot, bias=False)   # 槽位到槽位
        self.W_i2s = nn.Linear(d_intent, d_slot, bias=False) # 意图到槽位

        self.intent_layer_norm = nn.LayerNorm(d_intent)
        self.slot_layer_norm = nn.LayerNorm(d_slot)

        # 可选: 如果对 H_intent_aggregated / H_slot_aggregated 使用拼接，则需要融合层
        # 为简单起见，我们首先使用求和。如果使用拼接:
        # self.intent_fusion_linear = nn.Linear(d_intent * 2, d_intent)
        # self.slot_fusion_linear = nn.Linear(d_slot * 2, d_slot)

    def forward(
            self,
            h_intent_initial,
            h_slot_initial,
            affinity_matrix,
            padding_mask=None
    ):
        """
        前向传播函数。
        参数:
            H_intent_initial (torch.Tensor): 初始意图特征 (batch_size, seq_len, d_intent)
            H_slot_initial (torch.Tensor): 初始槽位特征 (batch_size, seq_len, d_slot)
            affinity_matrix (torch.Tensor): Token亲和力矩阵 (batch_size, seq_len, seq_len)。
                                          非对角线元素值在(0,1)之间，对角线为1。
            padding_mask (torch.Tensor, optional): 填充掩码 (batch_size, seq_len)。
                                                  Padding位置为True，有效token为False。
        返回:
            H_intent_final (torch.Tensor): 更新并融合后的意图特征
            H_slot_final (torch.Tensor): 更新并融合后的槽位特征
        """
        batch_size, seq_len, _ = h_intent_initial.shape

        # 1. (可选) 增强 affinity_matrix 对比度
        if self.affinity_power != 1.0:
            # affinity_matrix 没有零值且对角线为1.0。1.0的任何次方都是1.0。
            A_enhanced = torch.pow(affinity_matrix, self.affinity_power)
        else:
            A_enhanced = affinity_matrix

        # 2. 创建 A_masked，考虑padding
        # 这个 A_masked 会将涉及padding token的连接（及其自环）置零。
        A_masked = A_enhanced

        # 如果 padding_mask 不为 None，则会被定义，假定所有token都有效
        valid_token_mask = torch.ones(batch_size, seq_len, device=h_intent_initial.device)

        if padding_mask is not None:
            valid_token_mask = padding_mask.float()  # (B, N), 有效token为True
            # connectivity_mask: (B, N, N)。如果token i和j都有效，则为1，否则为0。
            connectivity_mask = valid_token_mask.unsqueeze(2).float() * valid_token_mask.unsqueeze(1)
            A_masked = A_enhanced * connectivity_mask
            # 对于一个padding token p, A_masked[b, p, p] 将变为 affinity_matrix[b,p,p] * 0 = 0
            # 对于一个有效token v, A_masked[b, v, v] 将变为 affinity_matrix[b,v,v] * 1 = 1 (因为原始对角线为1)

        # 3. 计算软度矩阵 D_soft 及其逆平方根 D_soft_inv_sqrt
        # 对 A_masked 的行求和。
        # 对于padding token，它们在 A_masked 中的行全是零，所以 D_diag 将为0。
        D_diag = torch.sum(A_masked, dim=-1)  # (B, N)

        # 在开方前加上epsilon，以防止孤立有效节点或padding节点导致除以零。
        D_diag_safe = D_diag + self.epsilon
        D_inv_sqrt_diag_values = 1.0 / torch.sqrt(D_diag_safe) # (B, N)

        if padding_mask is not None:
            # 对于padding token，它们的 D_inv_sqrt_diag_values 应该为0，
            # 这样它们就不会影响其他token的 A_norm 计算，
            # 并且它们在 A_norm 中对应的行/列也会变为零。
            D_inv_sqrt_diag_values = D_inv_sqrt_diag_values * valid_token_mask.float()

        D_soft_inv_sqrt = torch.diag_embed(D_inv_sqrt_diag_values)  # (B, N, N)

        # 4. 计算对称归一化的邻接矩阵 A_norm
        # A_norm = D^(-1/2) * A_masked * D^(-1/2)
        A_norm = D_soft_inv_sqrt @ A_masked @ D_soft_inv_sqrt
        # 对于padding token p, A_norm 的第 p 行和第 p 列将为零。

        # 5. GCN层信息传递
        # 意图节点更新
        transformed_intent_for_i2i = self.W_i2i(h_intent_initial) # (B, N, d_intent)
        transformed_slot_for_s2i = self.W_s2i(h_slot_initial)     # (B, N, d_intent)

        H_intent_from_intent = A_norm @ transformed_intent_for_i2i # (B, N, d_intent)
        H_intent_from_slot = A_norm @ transformed_slot_for_s2i   # (B, N, d_intent)

        # 融合来自不同来源的信息 (简单求和)
        H_intent_aggregated = H_intent_from_intent + H_intent_from_slot
        H_intent_updated = self.activation_fn(H_intent_aggregated)

        # 槽位节点更新
        transformed_slot_for_s2s = self.W_s2s(h_slot_initial)       # (B, N, d_slot)
        transformed_intent_for_i2s = self.W_i2s(h_intent_initial)   # (B, N, d_slot)

        H_slot_from_slot = A_norm @ transformed_slot_for_s2s     # (B, N, d_slot)
        H_slot_from_intent = A_norm @ transformed_intent_for_i2s # (B, N, d_slot)

        H_slot_aggregated = H_slot_from_slot + H_slot_from_intent
        H_slot_updated = self.activation_fn(H_slot_aggregated)

        # 6. 将padding mask应用于更新后的特征 (将padding位置置零)
        if padding_mask is not None:
            H_intent_updated = H_intent_updated * valid_token_mask.unsqueeze(-1).float()
            H_slot_updated = H_slot_updated * valid_token_mask.unsqueeze(-1).float()

        # 7. 特征融合 (显式残差连接)
        H_intent_final = h_intent_initial + H_intent_updated
        H_slot_final = h_slot_initial + H_slot_updated

        # 融合后可以添加 Layer Normalization
        H_intent_final = self.intent_layer_norm(H_intent_final)
        H_slot_final = self.slot_layer_norm(H_slot_final)

        return H_intent_final, H_slot_final

class GCNInteractionModule(nn.Module):
    def __init__(
            self, d_intent, d_slot,
            gcn_epsilon=1e-12,
            gcn_activation_fn_str="relu",
            gcn_affinity_power=1.0,
            num_gcn_layers=1
    ): # 添加了 num_gcn_layers
        """
        使用外部亲和力GCN的模型封装。
        参数:
            d_intent (int): 意图特征的维度。
            d_slot (int): 槽位特征的维度。
            gcn_epsilon (float): GCN层使用的epsilon。
            gcn_activation_fn_str (str): GCN层使用的激活函数名称。
            gcn_affinity_power (float): GCN层使用的亲和力幂次。
            num_gcn_layers (int): GCN层的数量 (堆叠)。
        """
        super().__init__()

        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(
                EnhancedAffinityGCNLayer(
                    d_intent, d_slot,
                    epsilon=gcn_epsilon,
                    activation_fn_str=gcn_activation_fn_str,
                    affinity_power=gcn_affinity_power
                )
            )

    def forward(
            self,
            h_intent_input,
            h_slot_input,
            affinity_matrix,
            padding_mask=None
    ):
        """
        前向传播函数。
        参数:
            h_intent_input (torch.Tensor): 初始意图特征 (batch_size, seq_len, d_intent)
            h_slot_input (torch.Tensor): 初始槽位特征 (batch_size, seq_len, d_slot)
            affinity_matrix (torch.Tensor): Token亲和力矩阵 (batch_size, seq_len, seq_len)
            padding_mask (torch.Tensor, optional): 填充掩码 (batch_size, seq_len)。Padding为True。
        """
        current_h_intent = h_intent_input
        current_h_slot = h_slot_input

        for gcn_layer in self.gcn_layers:
            current_h_intent, current_h_slot = gcn_layer(
                current_h_intent, current_h_slot, affinity_matrix, padding_mask
            )

        # H_intent_final = self.intent_final_ln(current_h_intent)
        # H_slot_final = self.slot_final_ln(current_h_slot)

        return current_h_intent, current_h_slot


# --- 示例用法 ---
if __name__ == '__main__':
    # 模型参数 (示例)
    d_intent_param = 64
    d_slot_param = 64
    gcn_affinity_power_param = 1.5 # 示例: 增强强连接
    num_gcn_layers_param = 2       # 示例: 堆叠2个GCN层

    model = GCNInteractionModule(
        d_intent=d_intent_param,
        d_slot=d_slot_param,
        gcn_affinity_power=gcn_affinity_power_param,
        num_gcn_layers=num_gcn_layers_param,
        gcn_activation_fn_str="gelu" # 示例使用 gelu
    )

    # 伪造输入数据
    batch_size = 4
    max_seq_len = 20 #批次中的最大序列长度

    # 模拟不同长度的序列
    seq_lens = [10, 15, max_seq_len, 12]

    # 创建填充后的输入特征
    hiddens_intent = torch.randn(batch_size, max_seq_len, d_intent_param)
    hiddens_slot = torch.randn(batch_size, max_seq_len, d_slot_param)

    # 创建padding_mask (True代表padding)
    padded_padding_mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool)
    for i, length in enumerate(seq_lens):
        padded_padding_mask[i, :length] = False # 有效token为False
        # 将输入中padding位置的特征置零 (好习惯)
        if length < max_seq_len:
            hiddens_intent[i, length:] = 0
            hiddens_slot[i, length:] = 0

    # 创建亲和力矩阵
    # 非对角线元素值在(0,1)之间，对角线为1。
    # 这适用于整个 (max_seq_len_eg, max_seq_len_eg) 矩阵。
    # GCN层将内部使用 padding_mask 来处理连接。
    dummy_affinity_values = torch.rand(batch_size, max_seq_len, max_seq_len)
    # 缩放以使非对角线元素主要小于1
    dummy_affinity_values = dummy_affinity_values * 0.8
    padded_affinity = torch.clamp(dummy_affinity_values, min=0.01, max=0.99) # 确保在(0,1)范围内
    # 将所有token（无论有效或padding）的对角线都设为1
    for b in range(batch_size):
        padded_affinity[b].fill_diagonal_(1.0)


    print("--- 输入形状 ---")
    print("填充后的 H_intent:", hiddens_intent.shape)
    print("填充后的 H_slot:", hiddens_slot.shape)
    print("填充后的 Affinity 矩阵:", padded_affinity.shape)
    print("填充后的 Padding Mask:", padded_padding_mask.shape)
    print("\n--- 运行模型 ---")

    # 前向传播
    final_intent_features, final_slot_features = model(
        hiddens_intent,
        hiddens_slot,
        padded_affinity,
        padding_mask=padded_padding_mask
    )

    print("\n--- 输出形状 ---")
    print("最终意图特征形状:", final_intent_features.shape)
    print("最终槽位特征形状:", final_slot_features.shape)

    # 验证: 检查输出中padding位置是否为零
    print("\n--- 验证输出中的Padding ---")
    padding_correct = True
    for i, length in enumerate(seq_lens):
        if length < max_seq_len:
            intent_padding_sum = torch.sum(torch.abs(final_intent_features[i, length:]))
            slot_padding_sum = torch.sum(torch.abs(final_slot_features[i, length:]))
            if intent_padding_sum > 1e-6 : # 使用一个小的容差来处理浮点精度问题
                print(f"意图样本 {i} 在位置 {length} 的Padding检查失败: 总和为 {intent_padding_sum.item()}")
                padding_correct = False
            if slot_padding_sum > 1e-6:
                print(f"槽位样本 {i} 在位置 {length} 的Padding检查失败: 总和为 {slot_padding_sum.item()}")
                padding_correct = False

    if padding_correct:
        print("所有样本的Padding置零检查通过。")
    else:
        print("部分样本的Padding置零检查失败。")

    print("\n--- 亲和力矩阵示例 (样本0, 前5x5) ---")
    print(padded_affinity[0, :5, :5])

    # 检查A_norm (需要更多访问权限或使用前向钩子)
    # 例如，如果你想检查第一层的A_norm:
    # A_norm_layer0 = model.gcn_layers[0].A_norm_debug # (如果你为了调试而存储它)
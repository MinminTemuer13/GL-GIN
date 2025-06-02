import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotDecoder(nn.Module):
    def __init__(self, hidden_dim, num_slot_labels, dropout_rate=0.1):
        """
         Args:
            hidden_dim (int): 输入词元编码的维度 (来自编码器)。
            num_slot_labels (int): 槽位标签的总数 (包括 'O' 标签)。
            dropout_rate (float): Dropout 比率。
        """
        super().__init__()
        self.num_slot_labels = num_slot_labels
        self.hidden_dim = hidden_dim

        # 线性层，用于将每个词元的编码映射到各个槽位标签的 Logits
        # 输出形状: (batch_size, max_seq_len, num_slot_labels)
        self.slot_logits_fc = nn.Linear(hidden_dim, num_slot_labels)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, token_encodings, padding_mask=None):
        """
        前向传播函数。

        Args:
            token_encodings (torch.Tensor): 一批词元的编码。
                                            形状: (batch_size, max_seq_len, hidden_dim)
            padding_mask (torch.Tensor, optional): 标记哪些是真实token，哪些是padding。
                                                   真实token为1，padding为0。
                                                   形状: (batch_size, max_seq_len)。
                                                   虽然对于逐个token的分类不是严格必需的（因为损失函数可以处理padding），
                                                   但在获取预测标签时可以用来忽略padding位置。

        Returns:
            tuple:
                - slot_logits (torch.Tensor): 每个词元对各个槽位标签的原始得分 (Logits)。
                                               形状: (batch_size, max_seq_len, num_slot_labels)
                - slot_probs (torch.Tensor): 每个词元对各个槽位标签的预测概率 (经过 Softmax)。
                                             形状: (batch_size, max_seq_len, num_slot_labels)
        """
        # 对词元编码应用 Dropout
        token_encodings_dropped = self.dropout(token_encodings)

        # 1. 计算每个词元对各个槽位标签的 Logits
        slot_logits = self.slot_logits_fc(token_encodings_dropped) # 形状: (batch_size, max_seq_len, num_slot_labels)

        # 2. 计算每个词元对各个槽位标签的概率 (用于推理或某些评估)
        slot_probs = F.softmax(slot_logits, dim=-1) # 形状: (batch_size, max_seq_len, num_slot_labels)

        # 注意：这里不需要像意图识别那样进行聚合或使用门控，
        # 因为槽位填充是词元级别的独立分类任务。
        # padding_mask 主要用于在计算损失或进行最终预测时忽略填充词元。

        return slot_logits, slot_probs
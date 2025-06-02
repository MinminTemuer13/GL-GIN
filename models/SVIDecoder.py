import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftVotingIntentDecoder(nn.Module):
    def __init__(self, hidden_dim, num_intents, dropout_rate=0.1):
        """
        Args:
            hidden_dim (int): 输入token编码的维度。
            num_intents (int): 可能的意图数量 (K)。
            dropout_rate (float): Dropout 比率。
        """
        super().__init__()
        self.num_intents = num_intents  # 意图数量
        self.hidden_dim = hidden_dim  # 隐藏层维度

        # 线性层，用于获取每个token对各个意图的倾向性 logits
        self.intent_tendency_fc = nn.Linear(hidden_dim, num_intents)

        # 线性层，用于获取每个token的显著性门控 logit
        self.significance_gate_fc = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout_rate)  # Dropout层

    def forward(self, token_encodings, padding_mask):
        """
        Args:
            token_encodings (torch.Tensor): 一批token的编码。
                                            形状: (batch_size, max_seq_len, hidden_dim)
            padding_mask (torch.Tensor): 标记哪些是真实token，哪些是padding。
                                         真实token为1，padding为0。
                                         形状: (batch_size, max_seq_len, 1)
        Returns:
            tuple:
                - sentence_intent_probs (torch.Tensor): 每个意图的预测概率（sigmoid之后）。
                                                       形状: (batch_size, num_intents)
                - sentence_intent_logits (torch.Tensor): 每个意图累积的分数（最终sigmoid之前）。
                                                        形状: (batch_size, num_intents)
                - token_intent_dist (torch.Tensor): 每个token的意图概率分布。
                                                   形状: (batch_size, max_seq_len, num_intents)
                - token_gates (torch.Tensor): 每个token的显著性门控值。
                                             形状: (batch_size, max_seq_len, 1)
        """
        # batch_size, max_seq_len, _ = token_encodings.shape # 获取批次大小和最大序列长度 (如果需要)
        token_encodings_dropped = self.dropout(token_encodings)  # 对token编码应用dropout

        # 1. Token级别的输出
        # 1.a. 意图倾向性 (dist_intent_t)
        intent_tendency_logits_t = self.intent_tendency_fc(token_encodings_dropped)
        token_intent_dist = F.softmax(intent_tendency_logits_t, dim=-1)

        # 1.b. 显著性门控 (gate_t)
        gate_logits_t = self.significance_gate_fc(token_encodings_dropped)
        token_gates = torch.sigmoid(gate_logits_t)

        # 2. 意图分数累积 (Score_j)
        weighted_dist_intent_t = token_intent_dist * token_gates
        # 应用掩码，确保 padding token 的贡献为 0
        # padding_mask 应该已经是 (batch_size, max_seq_len, 1) 并且是 float 类型
        masked_weighted_dist_intent_t = weighted_dist_intent_t * padding_mask
        sentence_intent_logits = torch.sum(masked_weighted_dist_intent_t, dim=1)

        # 3. 句子级别的意图预测 (P_sentence_intent_j)
        sentence_intent_probs = torch.sigmoid(sentence_intent_logits)

        return sentence_intent_logits, sentence_intent_probs, token_intent_dist, token_gates
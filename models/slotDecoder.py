import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

from models.components import SwiGLUFFN, RMSNorm


class SlotDecoder(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_slot_labels, dropout_rate, use_crf=True, eps_norm=1e-6): # 新增 use_crf 参数
        """
         Args:
            hidden_dim (int): 输入词元编码的维度 (来自编码器)。
            num_slot_labels (int): 槽位标签的总数 (包括 'O' 标签)。
            dropout_rate (float): Dropout 比率。
            use_crf (bool): 是否使用 CRF 层。
        """
        super().__init__()
        self.num_slot_labels = num_slot_labels
        self.hidden_dim = hidden_dim
        self.use_crf = use_crf # 保存 use_crf 状态
        self.dropout_rate = dropout_rate

        # 线性层，用于将每个词元的编码映射到各个槽位标签的 Logits (emissions for CRF)
        # 输出形状: (batch_size, max_seq_len, num_slot_labels)
        # 这个FC层现在输出的是 CRF 的 emission scores
        self.slot_emissions_ffn = nn.Sequential(
            RMSNorm(hidden_dim, eps=eps_norm),
            SwiGLUFFN(hidden_dim, ffn_dim, dropout_rate),
            nn.Dropout(dropout_rate)
        )

        self.slot_emissions_fc = nn.Sequential(
            nn.Linear(hidden_dim, num_slot_labels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )


        if self.use_crf:
            self.crf_layer = CRF(num_tags=num_slot_labels, batch_first=True)
        else:
            # 如果不使用CRF，我们仍然需要一个损失函数
            # 注意：损失函数的计算通常在主模型中进行，这里只是为了完整性，
            # 但更好的做法是将损失计算放到主模型的 forward 中。
            # 并且，ignore_index 应该从配置中传入，这里暂时写死为-100（PyTorch CrossEntropyLoss 默认的 ignore_index）
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # 假设 padding 标签 ID 是 -100

    def forward(self, token_encodings, slot_labels=None, attention_mask=None):
        """
        前向传播函数。
        Args:
            token_encodings (torch.Tensor): 一批词元的编码。
                                            形状: (batch_size, max_seq_len, hidden_dim)
            slot_labels (torch.Tensor, optional): 真实的槽位标签，用于训练时计算损失。
                                                  形状: (batch_size, max_seq_len)。
            attention_mask (torch.Tensor, optional): 标记哪些是真实token，哪些是padding。
                                                   真实token为1 (或 True)，padding为0 (或 False)。
                                                   形状: (batch_size, max_seq_len)。
                                                   对于CRF来说，这个mask非常重要。
        Returns:
            tuple:
                - slot_loss (torch.Tensor or None): 槽位填充的损失，训练时返回，推理时为 None。
                - slot_predictions (torch.Tensor or list):
                    - 如果不使用CRF (use_crf=False): 每个词元概率最高的标签索引。
                                                      形状: (batch_size, max_seq_len)
                    - 如果使用CRF (use_crf=True): 解码后的最优标签序列列表。
                                                   类型: list of lists of ints
        """
        # 1. 计算每个词元对各个槽位标签的 Emissions (Logits)
        # (batch_size, max_seq_len, num_slot_labels)
        slot_residual = token_encodings
        slot_ffn_output = self.slot_emissions_ffn(token_encodings)

        slot_emissions = self.slot_emissions_fc(slot_residual + slot_ffn_output)

        slot_loss = None
        slot_predictions = None

        if self.use_crf:
            if slot_labels is not None: # 训练模式
                # CRF 层需要 emissions, 真实标签, 以及掩码 (mask)
                # mask 告诉 CRF 哪些是真实的 token，哪些是 padding
                # attention_mask 通常可以直接用作 CRF 的 mask
                # CRF 的 forward 方法返回的是负对数似然损失
                slot_loss = -self.crf_layer(
                    emissions=slot_emissions,
                    tags=slot_labels.long(), # 确保标签是 LongTensor
                    mask=attention_mask, # 使用 bool 类型 mask
                    reduction='mean' # 或者 'token_mean'，或者不指定让其返回每个样本的loss再手动处理
                )
            else: # 推理模式
                # 使用 CRF 解码得到最优标签序列
                # decode 方法返回一个包含每个样本预测标签序列的列表
                # 每个序列是一个 Python list of ints
                slot_predictions = self.crf_layer.decode(
                    emissions=slot_emissions,
                    mask=attention_mask
                )
        else: # 不使用 CRF，使用逐词元的 CrossEntropyLoss
            if slot_labels is not None: # 训练模式
                # (batch_size, num_slot_labels, max_seq_len) for CrossEntropyLoss if batch_first=False
                # or (batch_size * max_seq_len, num_slot_labels)
                # CrossEntropyLoss expects:
                # Input: (N, C) or (N, C, d1, d2, ...) where C = number of classes
                # Target: (N) or (N, d1, d2, ...)
                active_loss = attention_mask.view(-1) == 1 # 只计算非padding部分的损失
                active_logits = slot_emissions.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]

                if active_logits.shape[0] > 0: # 确保有非填充词元
                    slot_loss = self.loss_fct(active_logits, active_labels.long())
                else:
                    slot_loss = torch.tensor(0.0, device=slot_emissions.device, requires_grad=True) # 确保可反向传播
            # 推理模式 (不使用CRF时)
            slot_probs = F.softmax(slot_emissions, dim=-1)
            slot_predictions = torch.argmax(slot_probs, dim=-1) # (batch_size, max_seq_len)

        return slot_loss, slot_predictions
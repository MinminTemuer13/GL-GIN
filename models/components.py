# --- SwiGLU 前馈网络 ---
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SwiGLUFFN(nn.Module):
    """
    SwiGLU 前馈网络。 FFN(x) = (Swish(x W_g) * (x W_1)) W_2
    """
    def __init__(self, d_model: int, d_ff: int, dropout_rate_ffn: float = 0.1):
        super().__init__()
        self.w_g = nn.Linear(d_model, d_ff) # 通常SwiGLU的门和主路径不加偏置
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout_ffn_internal = nn.Dropout(dropout_rate_ffn) # Dropout在最终输出前

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_val = F.silu(self.w_g(x))    # Swish(x W_g)
        hidden_val = self.w_1(x)          # x W_1
        return self.w_2(self.dropout_ffn_internal(gate_val * hidden_val))
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_in, n_layers=4, n_hidden_units=256):
        super().__init__()

        module_list = [nn.Linear(n_in, n_hidden_units), nn.ReLU(True)]
        for i in range(n_layers-1):
            if i != n_layers-2:
                # 除了最后一层，其他层都是输入大小和输出大小一致
                module_list += [nn.Linear(n_hidden_units, n_hidden_units),
                                nn.ReLU(True)]
            else:
                # 最后一层需要特别处理，输出大小为3
                # 最后一层输出是RGB，值在0~1,不能用ReLU
                module_list += [nn.Linear(n_hidden_units, 3),
                                nn.Sigmoid()]

        self.net = nn.Sequential(*module_list)

    def forward(self, x):
        """
        x: (B, 2) # pixel uv (normalized)
        """
        return self.net(x)  # (B, 3) rgb


class PositionalEncoding(nn.Module):
    def __init__(self, P):
        super().__init__()
        self.P = P

    def forward(self, x):
        # 计算 x 与 self.P 的转置的矩阵乘积。
        return torch.cat([torch.sin(x @ self.P.T)])

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_layers=4, n_hidden_units=256):
        super().__init__()
        module_list = [nn.Linear(2, n_hidden_units), nn.ReLU(True)]
        for i in range(n_layers-1):
            if i != n_layers-2:
                # 除了最后一层，其他层都是输入大小和输出大小一致
                module_list += [nn.Linear(n_hidden_units,
                                          n_hidden_units), nn.ReLU(True)]
            else:
                # 最后一层需要特别处理，输出大小为3
                module_list += [nn.Linear(n_hidden_units, 3), nn.ReLU(True)]
        self.module = nn.Sequential(*module_list)

    def forward(self, x):
        return self.module(x)

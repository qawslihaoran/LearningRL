import torch
import torch.nn as nn
import torch.nn.functional as F


class PGNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """
        初始化q网络,为全连接网络
        :param input_dim: 输入特征数,即环境的状态维数
        :param output_dim: 输出的动作维数
        :param hidden_dim: 隐藏层维数
        """
        super(PGNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu((self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x

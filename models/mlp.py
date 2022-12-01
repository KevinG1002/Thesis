import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Modules):
    """
    Basic MLP class for experiments
    """

    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        self.fc_1 = nn.Linear(input_dim, 128)
        self.fc_2 = nn.Linear(512, 64)
        self.fc_3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        out = F.softmax(x)
        return out

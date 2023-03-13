import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class MLP(nn.Module):
    """
    Basic MLP class for experiments. 4 layers including input and output layers.
    """

    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.name = self._get_name()
        self.fc_1 = nn.Linear(input_dim, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        out = F.softmax(x)
        return out


class SmallMLP(nn.Module):
    """
    Basic MLP class for experiments. 3 layers including input and output layers.
    """

    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, 50)
        self.fc_2 = nn.Linear(50, num_classes)

        self.name = self._get_name()

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        out = F.softmax(x)
        return out


class RegressMLP(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int = 1):
        """
        Mini MLP for Sine Regression. Here, the activation function is ReLU.
        """
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, 10)
        self.fc_2 = nn.Linear(10, output_dim)
        self.name = self._get_name()

    def forward(self, x):
        x = F.tanh(self.fc_1(x))
        x = F.tanh(self.fc_2(x))
        # out = F.softmax(x)
        return x


class RegressMLPTwo(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int = 1):
        """
        Mini MLP for Sine Regression Task. Here, the activation function is Sigmoid.
        """
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, 10)
        self.fc_2 = nn.Linear(10, output_dim)

        self.name = self._get_name()

    def forward(self, x):
        x = F.sigmoid(self.fc_1(x))
        x = F.sigmoid(self.fc_2(x))
        out = F.softmax(x)
        return out


def test():
    mlp = MLP()
    print(mlp)
    print(summary(model=mlp, input_size=(784,)))


if __name__ == "__main__":
    test()

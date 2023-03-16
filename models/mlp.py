import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary
import copy

torch.manual_seed(0)


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
    Basic MLP class for experiments. 2 layers including input and output layers.
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
    from datasets.get_dataset import DatasetRetriever
    from torch.utils.data import DataLoader

    d = DatasetRetriever("MNIST")
    train_set, _ = d()
    mlp = SmallMLP()
    dataloader = DataLoader(train_set, 64, shuffle=False)
    print(mlp)
    # print(summary(model=mlp, input_size=(784,)))
    optim1 = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    mlp_2 = copy.deepcopy(mlp)
    optim2 = torch.optim.Adam(mlp_2.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    print("Original Train Loop")
    # for idx, (mbatchx, mbatchy) in enumerate(dataloader):
    #     optim1.zero_grad()
    #     flattened_mbatch_x = torch.flatten(mbatchx, start_dim=1)
    #     one_hot_mbatch_y = torch.eye(10)[mbatchy]
    #     pred_y = mlp(flattened_mbatch_x)
    #     loss = loss_fn(
    #         pred_y,
    #         one_hot_mbatch_y,
    #     )
    #     loss.backward()
    #     print(f"Batch size 1 (batch {idx}) - grad: {mlp.fc_2.weight.grad[0][0:10]}")
    #     print(f"Batch size 1 (batch {idx}) - weight: {mlp.fc_2.weight[0][0:10]}")
    #     optim1.step()
    #     if idx == 3:
    #         exit()

    # for idx, (mbatchx, mbatchy) in enumerate(dataloader):
    #     flattened_mbatch_x = torch.flatten(mbatchx, start_dim=1)
    #     one_hot_mbatch_y = torch.eye(10)[mbatchy]
    #     pred_y = mlp_2(flattened_mbatch_x)
    #     loss = (
    #         loss_fn(
    #             pred_y,
    #             one_hot_mbatch_y,
    #         )
    #         / 3
    #     )
    #     loss.backward()
    #     grads = mlp_2.fc_1.weight.grad
    #     print(f"Batch {idx} - grad: {mlp_2.fc_2.weight.grad[0][0:10]}")
    #     print(f"Batch {idx} - weight: {mlp_2.fc_2.weight[0][0:10]}")
    #     if (idx + 1) % 3 == 0:
    #         optim2.step()
    #         optim2.zero_grad()
    #         print("\nWeights updated")
    #     if idx == 10:
    #         exit()


if __name__ == "__main__":
    test()

import sys
import torch
from torchvision.datasets import MNIST
from torchvision import transforms

sys.path.append("..")
from models.mlp import MLP
from templates.sgd_template import SupervisedLearning

torch.manual_seed(17)


def run():
    mnist_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )  # potentially add more transforms
    train_set = MNIST(
        root="../datasets", train=True, download=True, transform=mnist_transforms
    )
    test_set = MNIST(
        root="../datasets", train=False, download=True, transform=mnist_transforms
    )

    supervised_learning_process = SupervisedLearning(
        MLP(), train_set, test_set, None, 10, 20, 64
    )

    supervised_learning_process.train()
    supervised_learning_process.test()


if __name__ == "__main__":
    run()

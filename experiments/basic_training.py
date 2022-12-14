import sys
import torch
import os
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss
from torch.optim import Adam

sys.path.append("..")
from models.mlp import MLP
from frameworks.sgd_template import SupervisedLearning
from utils.params import argument_parser

torch.manual_seed(17)


class CONFIG:
    def __init__(
        self,
        model: nn.Module,
        dataset_dir: str = "../datasets/MNIST",
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.999,
        optimizer: torch.optim.Optimizer = None,
        loss_function: _Loss = None,
        im_transforms: transforms = None,
    ):
        self.model = model
        self.dataset_dir = dataset_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = (
            optimizer
            if optimizer
            else Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        )
        self.loss_function = loss_function if loss_function else CrossEntropyLoss()

        self.im_transforms = im_transforms if im_transforms else transforms.ToTensor()
        if os.path.dirname(self.dataset_dir) == "MNIST":
            self.train_set = MNIST(
                self.dataset_dir,
                train=True,
                transforms=self.im_transforms,
            )
            self.test_set = MNIST(
                self.dataset_dir,
                train=True,
                transforms=self.im_transforms,
            )
        elif os.path.dirname(self.dataset_dir) == "CIFAR10":
            self.train_set = CIFAR10(
                self.dataset_dir,
                train=True,
                transforms=self.im_transforms,
            )
            self.test_set = CIFAR10(
                self.dataset_dir,
                train=True,
                transforms=self.im_transforms,
            )
        else:
            raise NotImplementedError


def run(cfg: CONFIG):
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

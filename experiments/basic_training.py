import sys
import torch
import copy
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
        num_classes: int = 10,
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
        self.num_classes = num_classes
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
        if os.path.basename(self.dataset_dir) == "MNIST":
            self.train_set = MNIST(
                os.path.join(self.dataset_dir),
                train=True,
                transform=self.im_transforms,
                download=True,
            )
            self.test_set = MNIST(
                self.dataset_dir,
                train=True,
                transform=self.im_transforms,
                download=True,
            )
        elif os.path.basename(self.dataset_dir) == "CIFAR10":
            self.train_set = CIFAR10(
                self.dataset_dir,
                train=True,
                transform=self.im_transforms,
                download=True,
            )
            self.test_set = CIFAR10(
                self.dataset_dir,
                train=True,
                transform=self.im_transforms,
                download=True,
            )
        else:
            raise NotImplementedError


def run_basic_training(cfg: CONFIG):
    supervised_learning_experiment = SupervisedLearning(
        model=copy.deepcopy(cfg.model),
        train_set=cfg.train_set,
        test_set=cfg.test_set,
        val_set=None,
        num_classes=cfg.num_classes,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        optim=cfg.optimizer,
        criterion=cfg.loss_function,
    )

    supervised_learning_experiment.train()
    performance_dict = supervised_learning_experiment.test()
    print(performance_dict["test_loss"].numpy())
    trained_model = copy.deepcopy(cfg.model)
    return trained_model, performance_dict


def main():
    model = MLP()
    loss_func = CrossEntropyLoss()
    params = argument_parser()
    optimizer = Adam(
        model.parameters(), params.learning_rate, weight_decay=params.weight_decay
    )
    cfg = CONFIG(
        model,
        params.dataset,
        10,
        params.num_epochs,
        params.batch_size,
        params.learning_rate,
        params.weight_decay,
        optimizer,
        loss_func,
    )
    run_basic_training(cfg)


if __name__ == "__main__":
    main()

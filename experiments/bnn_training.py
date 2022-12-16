import sys
import torch
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss
import torch.nn as nn
from torch.distributions import Uniform
import os

sys.path.append("..")

from frameworks.vi_template import VITemplate
from models.bnn import SimpleBNN
from distributions.gaussians import *
from distributions.laplace import LaPlaceDistribution
from distributions.uniform import UniformDistribution
from utils.params import *


class CONFIG:
    def __init__(
        self,
        model: nn.Module,
        dataset_dir: str = "../datasets/MNIST",
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.999,
        num_mc_samples: int = 25,
        optimizer: torch.optim.Optimizer = None,
        loss_function: _Loss = None,
        prior_dist: ParameterDistribution = None,
        im_transforms: transforms = None,
    ):
        self.model = model
        self.dataset_dir = dataset_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_mc_samples = num_mc_samples
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
        self.prior_dist = prior_dist if prior_dist else UnivariateGaussian(0, 1)
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


def run_bnn_training(cfg: CONFIG):

    vi_experiment = VITemplate(
        model=cfg.model,
        train_set=cfg.train_set,
        test_set=cfg.test_set,
        val_set=None,
        num_classes=len(cfg.train_set.classes),
        batch_size=cfg.batch_size,
        num_mc_samples=cfg.num_mc_samples,
        epochs=cfg.epochs,
        optim=cfg.optimizer,
        likelihood_criterion=cfg.loss_function,
        device=cfg.device,
    )

    vi_experiment.train()
    vi_experiment.evaluate()


def main():
    bnn = SimpleBNN(
        784,
        10,
        UnivariateGaussian(0, 0.5),
    )
    params = argument_parser()

    cfg = CONFIG(
        model=bnn,
        dataset_dir=params.dataset,
        epochs=params.num_epochs,
        batch_size=params.batch_size,
        learning_rate=params.learning_rate,
        weight_decay=params.weight_decay,
    )
    run_bnn_training(cfg)


if __name__ == "__main__":
    main()
    # print(torch.log(torch.exp(torch.tensor(-2))))

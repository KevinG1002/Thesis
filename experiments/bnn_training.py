import sys
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss

sys.path.append("..")

from frameworks.vi_template import VITemplate
from models.bnn import SimpleBNN
from distributions.gaussians import *
from distributions.laplace import LaPlaceDistribution


def run():

    train_set = MNIST(
        root="../datasets",
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )
    test_set = MNIST(
        root="../datasets",
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )

    bnn = SimpleBNN(
        784,
        len(train_set.classes),
        UnivariateGaussian(0, 4),
    )

    vi_experiment = VITemplate(
        model=bnn,
        train_set=train_set,
        test_set=test_set,
        val_set=None,
        num_classes=len(train_set.classes),
        batch_size=128,
        num_mc_samples=30,
        epochs=50,
        optim=Adam(bnn.parameters(), 0.001, weight_decay=1e-3),
        likelihood_criterion=CrossEntropyLoss(reduction="sum"),
        device="cpu",
    )

    vi_experiment.train()
    vi_experiment.evaluate()


if __name__ == "__main__":
    run()
    # print(torch.log(torch.exp(torch.tensor(-2))))

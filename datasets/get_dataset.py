import torch
from torchvision.datasets import MNIST, CIFAR10, CelebA
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class DatasetRetriever:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def __call__(self) -> "tuple[Dataset, Dataset]":
        if self.dataset_name == "MNIST":
            im_transforms = transforms.ToTensor()
            self.train_set = MNIST(
                "../datasets/",
                train=True,
                transform=im_transforms,
            )
            self.test_set = MNIST(
                "../datasets/",
                train=False,
                transform=im_transforms,
            )
            return self.train_set, self.test_set

        elif self.dataset_name == "CIFAR10":
            im_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),
                ]
            )
            self.train_set = CIFAR10(
                "../datasets/",
                train=True,
                transform=im_transforms,
            )
            self.test_set = CIFAR10(
                "../datasets/",
                train=False,
                transform=im_transforms,
            )
            return self.train_set, self.test_set

        elif self.dataset_name == "CelebA":
            im_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),
                ]
            )
            self.train_set = CelebA(
                "../datasets/",
                split="train",
                target_type="identity",
                transform=im_transforms,
            )
            self.test_set = CelebA(
                "../datasets/",
                split="test",
                target_type="identity",
                transform=im_transforms,
            )
            return self.train_set, self.test_set

        else:
            raise "Dataset not available yet"

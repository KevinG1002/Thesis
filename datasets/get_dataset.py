import torch
from torchvision.datasets import MNIST, CIFAR10, CelebA
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class DatasetRetriever:
    def __init__(
        self,
        dataset_name: str,
        resize_option: bool = False,
        resize_dim: tuple = (None, None),
    ):
        self.dataset_name = dataset_name
        self.resize_option = resize_option
        if self.resize_option:
            self.resize_dim = resize_dim

        self.im_transforms = {
            "scale": transforms.ToTensor(),
            "normalize": transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
            ),
            "resize": transforms.Resize(size=resize_dim),
        }

    def __call__(self) -> "tuple[Dataset, Dataset]":
        if self.dataset_name == "MNIST":
            if self.resize_option:
                im_transforms = (
                    transforms.Compose(
                        [
                            v
                            for k, v in self.im_transforms.items()
                            if k in ["scale", "resize"]
                        ]
                    ),
                )[0]

                print(im_transforms)
                self.train_set = MNIST(
                    "../datasets/", train=True, transform=im_transforms
                )
                self.test_set = MNIST(
                    "../datasets/",
                    train=False,
                    transform=im_transforms,
                )
                return self.train_set, self.test_set

            im_transforms = self.im_transforms["scale"]
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
            if self.resize_option:
                im_transforms = transforms.Compose(list(self.im_transforms.values()))
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

            im_transforms = transforms.Compose(
                [
                    v
                    for k, v in self.im_transforms.items()
                    if k in ["scale", "normalize"]
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
            if self.resize_option:
                im_transforms = transforms.Compose(list(self.im_transforms.values()))
                self.train_set = CelebA(
                    "../datasets/",
                    train=True,
                    transform=im_transforms,
                )
                self.test_set = CelebA(
                    "../datasets/",
                    train=False,
                    transform=im_transforms,
                )
                return self.train_set, self.test_set

            im_transforms = transforms.Compose(
                [
                    v
                    for k, v in self.im_transforms.items()
                    if k in ["scale", "normalize"]
                ]
            )
            self.train_set = CelebA(
                "../datasets/",
                train=True,
                transform=im_transforms,
            )
            self.test_set = CelebA(
                "../datasets/",
                train=False,
                transform=im_transforms,
            )
            return self.train_set, self.test_set

        else:
            raise "Dataset not available yet"

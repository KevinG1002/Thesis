import torch
import copy
import json
import torch.nn as nn
import sys
import os
from params import *
from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.optim import Adam

sys.path.append("../")
from experiments.basic_training import run_basic_training, CONFIG
from models.mlp import MLP
from frameworks.sgd_template import SupervisedLearning


class GENCONFIG:
    def __init__(
        self,
        num_runs: int = 300,
        target_dataset_path: str = None,
        target_dataset_transforms: transforms = transforms.ToTensor(),
        target_dir: str = None,
    ):
        assert target_dir, "Target directory for trained models needs to be provided."
        assert (
            target_dataset_path
        ), "Target dataset path needs to be provided to train collection of models for dataset."
        self.num_runs = num_runs
        self.target_dir = target_dir
        self.target_dataset_path = target_dataset_path
        self.target_dataset_transforms = target_dataset_transforms
        if os.path.basename(self.target_dataset_path) == "MNIST":
            self.train_set = MNIST(
                os.path.join(self.target_dataset_path),
                train=True,
                transform=self.target_dataset_transforms,
                download=True,
            )
            self.test_set = MNIST(
                self.target_dataset_path,
                train=False,
                transform=self.target_dataset_transforms,
                download=True,
            )
        elif os.path.basename(self.target_dataset_path) == "CIFAR10":
            self.train_set = CIFAR10(
                self.target_dataset_path,
                train=True,
                transform=self.target_dataset_transforms,
                download=True,
            )
            self.test_set = CIFAR10(
                self.target_dataset_path,
                train=False,
                transform=self.target_dataset_transforms,
                download=True,
            )
        else:
            raise NotImplementedError


# def reset_parameters(model: nn.Module):
#     for layer in model.children():
#         if hasattr(layer, "reset_parameters"):
#             layer.reset_parameters()

#     return model


def run(cfg: GENCONFIG):
    if not os.path.exists(os.path.join(cfg.target_dir, "models/")):
        os.mkdir(f"{cfg.target_dir}/models/")

    dataset_dict = {}
    params = argument_parser()

    models = [MLP() for _ in range(cfg.num_runs)]
    optimizers = [
        Adam(models[i].parameters(), lr=params.learning_rate)
        for i in range(cfg.num_runs)
    ]
    for i in range(cfg.num_runs):
        model_path = f"{cfg.target_dir}models/mlp_mnist_model_{i}.pth"
        print("\nModel Dataset Gen Model object id", id(models[i]))
        print("Model Dataset Gen Optimizer object id", id(optimizers[i]))
        training_process = SupervisedLearning(
            model=models[i],
            train_set=cfg.train_set,
            test_set=cfg.test_set,
            val_set=None,
            num_classes=10,
            epochs=params.num_epochs,
            batch_size=params.batch_size,
            optim=optimizers[i],
            criterion=CrossEntropyLoss(),
        )
        training_process.train()
        performance_dict = training_process.test()
        trained_model = copy.deepcopy(training_process.model)
        # trained_model, performance_dict = run_basic_training(run_configs[i])
        torch.save(trained_model.state_dict(), model_path)
        dataset_dict[model_path] = float(performance_dict["test_loss"].numpy())

    return dataset_dict


def main():
    target_dataset = "MNIST"
    target_directory = f"../datasets/model_dataset_{target_dataset}/"
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)

    cfg = GENCONFIG(
        num_runs=10,
        target_dir=target_directory,
        target_dataset_transforms=transforms.ToTensor(),
        target_dataset_path="../datasets/MNIST",
    )
    dataset_dict = run(cfg)
    with open(f"{target_directory}/model_dataset.json", "w") as file:
        json.dump(dataset_dict, file)


if __name__ == "__main__":
    main()

import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10
from frameworks.sgd_template import SupervisedLearning
from datasets.model_dataset import ModelsDataset
from datasets.get_dataset import DatasetRetriever
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from frameworks.ddpm_template import DDPMDiffusion
from models.mlp import MLP
from utils.weight_transformations import nn_to_2d_tensor, tensor_to_nn


class CONFIG:
    def __init__(
        self,
        dataset_name: Dataset,
        n_diffusion_steps: int,
        sample_channels: int,
        num_channels: int = 64,
        sample_size: "tuple[int]" = (24, 24),
        channel_mulipliers: "list[int]" = [1, 2, 2, 4],
        is_attention: "list[bool]" = [False, False, False, False],
        batch_size: int = 16,
        epochs: int = 50,
        learning_rate: float = 1e-4,
        n_samples_gen: int = 5,
    ):
        self.n_diffusion_steps = n_diffusion_steps
        self.num_channels = num_channels
        self.sample_channels = sample_channels
        self.sample_size = sample_size
        self.channel_multipliers = channel_mulipliers
        self.is_attention = is_attention
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n_samples_gen = n_samples_gen

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if dataset_name == "MNIST":
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize(self.sample_size)]
            )
            self.dataset = MNIST(
                "../datasets/",
                train=True,
                transform=self.transforms,
                download=True,
            )
        elif dataset_name == "CIFAR10":
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize(self.sample_size)]
            )
            self.dataset = CIFAR10(
                "../datasets/",
                train=True,
                transform=self.transforms,
                download=True,
            )
        elif dataset_name == "model_dataset_MNIST":
            self.dataset = ModelsDataset(
                f"../datasets/{dataset_name}",
                f"../datasets/{dataset_name}/model_dataset.json",
                MLP(),
                nn_to_2d_tensor,
            )


def run(cfg: CONFIG):
    diffusion_process = DDPMDiffusion(
        diffusion_steps=cfg.n_diffusion_steps,
        sample_channels=1,
        num_channels=cfg.num_channels,
        sample_dimensions=cfg.sample_size,
        channel_multipliers=cfg.channel_multipliers,
        is_attention=cfg.is_attention,
        num_gen_samples=cfg.n_samples_gen,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        epochs=cfg.epochs,
        dataset=cfg.dataset,
        device=cfg.device,
    )
    print("Experiment config: %s" % diffusion_process.__dict__)
    if not os.path.exists("../checkpoints"):
        os.mkdir("../checkpoints")
    model_path = "../checkpoints/ddpm.pt"
    diffusion_process.train()
    torch.save(
        {
            "epochs": cfg.epochs,
            "unet_state_dict": diffusion_process.noise_predictor.state_dict(),
            "optimizer_state_dict": diffusion_process.optimizer.state_dict(),
            "loss": diffusion_process.loss,
        },
        model_path,
    )
    (
        sample_1,
        sample_2,
        sample_3,
        sample_4,
        sample_5,
    ) = diffusion_process.sample()

    generated_model_1 = tensor_to_nn(sample_1, cfg.dataset.base_model)

    _, test_set = DatasetRetriever(cfg.dataset.original_dataset)
    test_process = SupervisedLearning(generated_model_1, test_set=test_set)
    test_process.test()


def main():
    dataset_name = "model_dataset_MNIST"
    cfg = CONFIG(dataset_name, 100, 1, epochs=1, batch_size=2, sample_size=(368, 320))
    run(cfg)


if __name__ == "__main__":
    main()

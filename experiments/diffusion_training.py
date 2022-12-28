import torch, sys
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

sys.path.append("../")
from frameworks.ddpm_template import DDPMDiffusion


class CONFIG:
    def __init__(
        self,
        dataset_name: Dataset,
        n_diffusion_steps: int,
        sample_channels: int,
        num_channels: int = 64,
        sample_size: tuple[int] = (24, 24),
        channel_mulipliers: list[int] = [1, 2, 2, 4],
        is_attention: list[bool] = [False, False, False, False],
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
        else:
            raise NotImplementedError


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

    diffusion_process.train()
    diffusion_process.sample()


def main():
    dataset_name = "MNIST"
    cfg = CONFIG(dataset_name, 100, 1)
    run(cfg)


if __name__ == "__main__":
    main()

import torch
import os
import datetime
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10
from datasets.get_dataset import DatasetRetriever
from frameworks.ddpm_template import DDPMDiffusion
from models.mlp import MLP
from utils.weight_transformations import nn_to_2d_tensor, tensor_to_nn
from utils.logging import Logger

EXPERIMENTAL_RESULTS_PATH = "experimental_results"


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
        log_training: bool = False,
        checkpoint_every: int = None,
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
        self.dataset_name = dataset_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dataset = DatasetRetriever(
            self.dataset_name, resize_option=True, resize_dim=self.sample_size
        )
        self.dataset, _ = dataset()
        self.experiment_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_DDPM_{self.dataset_name}_e_{self.epochs}_{self.n_diffusion_steps}_steps"
        self.log_training = log_training
        # Checkpointing attributes; if log_training is False, path stay at None and checkpoint every does't change.
        self.checkpoint_path = None
        self.checkpoint_every = checkpoint_every
        self.experiment_config = {
            k: v
            for k, v in self.__dict__.items()
            if type(v) in [str, int, float, bool, tuple, list]
        }
        print("\nExperiment Config:\n%s" % self.experiment_config)
        if self.log_training:
            if os.path.isdir(EXPERIMENTAL_RESULTS_PATH):
                self.experiment_dir = os.path.join(
                    EXPERIMENTAL_RESULTS_PATH, self.experiment_name
                )
                if not os.path.exists(self.experiment_dir):
                    os.mkdir(self.experiment_dir)

                self.logger = Logger(
                    self.experiment_name, self.experiment_dir, self.experiment_config
                )
            else:
                os.mkdir(EXPERIMENTAL_RESULTS_PATH)
                self.experiment_dir = os.path.join(
                    EXPERIMENTAL_RESULTS_PATH, self.experiment_name
                )
                if not os.path.exists(self.experiment_dir):
                    os.mkdir(self.experiment_dir)

                self.logger = Logger(
                    self.experiment_name, self.experiment_dir, self.experiment_config
                )
            self.checkpoint_path = (
                self.logger.checkpoint_path
            )  # automatically generated with logger object.


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
        checkpoint_every=cfg.checkpoint_every,
        checkpoint_dir_path=cfg.checkpoint_path,
    )

    train_metrics = diffusion_process.train()
    if cfg.log_training:
        cfg.logger.save_results(train_metrics)
    (
        sample_1,
        sample_2,
        sample_3,
        sample_4,
        sample_5,
    ) = diffusion_process.sample()

    # generated_model_1 = tensor_to_nn(sample_1, cfg.dataset.base_model)

    # _, test_set = DatasetRetriever(cfg.dataset.original_dataset)
    # test_process = SupervisedLearning(generated_model_1, test_set=test_set)
    # test_process.test()


def main():
    dataset_name = "MNIST"
    cfg = CONFIG(
        dataset_name,
        10,
        1,
        epochs=1,
        batch_size=32,
        sample_size=(24, 24),
        log_training=True,
        checkpoint_every=1,
    )
    run(cfg)


if __name__ == "__main__":
    main()

import torch
import os
import datetime
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10
from datasets.get_dataset import DatasetRetriever
from frameworks.ddpm_template import DDPMDiffusion
from frameworks.sgd_template import SupervisedLearning
from models.mlp import MLP
from models.diffusion_sampler import DDPMSampler
from utils.weight_transformations import nn_to_2d_tensor, tensor_to_nn
from utils.exp_logging import Logger

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
        n_blocks: int = 2,
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
        self.n_blocks = n_blocks
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n_samples_gen = n_samples_gen
        self.dataset_name = dataset_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dataset = DatasetRetriever(
            self.dataset_name, resize_option=False, resize_dim=self.sample_size
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
        n_blocks=cfg.n_blocks,
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

    diffusion_process.sample("after_training")

    diffusion_sampler = DDPMSampler(
        diffusion_process,
        sample_channels=cfg.sample_channels,
        sample_size=cfg.sample_size,
        device=cfg.device,
        checkpoint_dir=cfg.checkpoint_path,
    )
    diffusion_sampler.sample()
    ## DIFFUSION MODEL ON MLP DATASET ##
    # checkpoint = torch.load("/scratch_net/bmicdl03/kgolan/Thesis/experiments/experimental_results/2023-01-12_11-23-54_DDPM_model_dataset_MNIST_e_1_150_steps/checkpoints/ddpm_fully_trained_e_1_loss_1.000.pt")
    # diffusion_process.checkpoint_dir_path = "/scratch_net/bmicdl03/kgolan/Thesis/experiments/experimental_results/2023-01-12_11-23-54_DDPM_model_dataset_MNIST_e_1_150_steps/checkpoints/"
    # diffusion_process.noise_predictor.load_state_dict(checkpoint["unet_state_dict"])
    gen_samples = diffusion_process.sample("after_training")

    gen_model_test_dataset = DatasetRetriever(cfg.dataset.original_dataset)
    _, test_set = gen_model_test_dataset()

    generated_model_1 = tensor_to_nn(gen_samples[0], cfg.dataset.base_model)
    test_process = SupervisedLearning(
        generated_model_1, test_set=test_set, device=cfg.device
    )
    print("\nTesting Generated Sample 1:\n")
    test_process.test()

    generated_model_2 = tensor_to_nn(gen_samples[1], cfg.dataset.base_model)
    test_process_2 = SupervisedLearning(
        generated_model_2, test_set=test_set, device=cfg.device
    )
    print("\nTesting Generated Sample 2:\n")
    test_process_2.test()

    generated_model_3 = tensor_to_nn(gen_samples[2], cfg.dataset.base_model)
    test_process_3 = SupervisedLearning(
        generated_model_3, test_set=test_set, device=cfg.device
    )
    print("\nTesting Generated Sample 3:\n")
    test_process_3.test()

    generated_model_4 = tensor_to_nn(gen_samples[3], cfg.dataset.base_model)
    test_process_4 = SupervisedLearning(
        generated_model_4, test_set=test_set, device=cfg.device
    )
    print("\nTesting Generated Sample 4:\n")
    test_process_4.test()

    generated_model_5 = tensor_to_nn(gen_samples[4], cfg.dataset.base_model)
    test_process_5 = SupervisedLearning(
        generated_model_5, test_set=test_set, device=cfg.device
    )
    print("\nTesting Generated Sample 5:\n")
    test_process_5.test()


def main():
    # dataset_name = "MNIST"
    dataset_name = "model_dataset_MNIST"
    cfg = CONFIG(
        dataset_name,
        n_diffusion_steps=1000,
        sample_channels=1,
        epochs=1,
        learning_rate=2e-5,
        batch_size=3,
        sample_size=(None, None),
        log_training=True,
        checkpoint_every=1,
        n_blocks=2,
        is_attention=[False, False, False, True],
    )
    run(cfg)


if __name__ == "__main__":
    main()

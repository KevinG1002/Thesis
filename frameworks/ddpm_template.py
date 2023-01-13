import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from datasets.model_dataset import ModelsDataset
from models.ddpm import DDPM
from utils.exp_logging import checkpoint
from models.unet import DDPMUNet


class DDPMDiffusion:
    """
    DDPM Diffusion class.
    - ddpm: The actual diffusion model that will learn the underlying probability distribution of the dataset you want to synthesize samples from.
    - noise_predictor: model that learns to predict the noise added to a sample durring forward diffusion.
    """

    ddpm: DDPM
    noise_predictor: DDPMUNet
    optimizer: Adam

    def __init__(
        self,
        diffusion_steps: int,
        sample_channels: int,
        num_channels: int,
        sample_dimensions: "tuple[int]",
        channel_multipliers: "list[int]",
        is_attention: "list[bool]",
        n_blocks: int,
        num_gen_samples: int,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        dataset: Dataset,
        device: str,
        checkpoint_every: int,
        checkpoint_dir_path: str,
    ) -> None:
        """
        Diffusion Model template for training and generation of samples belonging to some dataset.

        This template takes two models as input: the DDPM (Denoising Diffusion Probabilistic Model)
        and a Noise Predictor, sometimes known as "epsilon model", that predicts the noise that has
        been added to samples distorted by the diffusion process.

        A nice explanation for how these two need to interact with each other can be found here:
        https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

        The following is a summary of the necessary arguments for this process to run.
        Args:
            - sample_channels: number of channels in a given sample in your dataset.
            - num_channels: number of channels for first projection of sample in UNet
            - sample_dimensions: shape of a given sample passed to the model.
            - channel_multipliers: list denoting how to multiply channels at different resolutions in UNet
            - is_attention: list of booleans denoting if an attention block should be used at a resolution level in a UNet.
            - num_gen_samples: number of samples to generate once diffusion process is done.
            - batch_size: batch_size during training of UNet model
            - learning_rate: learning rate for UNEt optimizer
            - epochs: number of epochs for UNet to iterate over dataset.
            - dataset: dataset to synthesize examples from.


        """
        self.diffusion_steps = diffusion_steps
        self.sample_channels = sample_channels
        self.num_channels = num_channels
        self.sample_dimensions = sample_dimensions
        self.channel_multipliers = channel_multipliers
        self.is_attention = is_attention
        self.num_gen_samples = num_gen_samples
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset = dataset
        self.device = device
        self.noise_predictor = DDPMUNet(
            sample_channels, num_channels, channel_multipliers, is_attention, n_blocks
        ).to(self.device)
        self.ddpm = DDPM(self.noise_predictor, self.diffusion_steps, self.device)
        self.optimizer = Adam(self.noise_predictor.parameters(), self.learning_rate)

        # Checkpointing attributes
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir_path = checkpoint_dir_path
        assert (checkpoint_dir_path and checkpoint_every) or not (
            checkpoint_dir_path and checkpoint_every
        ), "Missing either one of checkpoint dir (str path) or checkpoint frequency (int)"
        if checkpoint_every:
            assert (
                checkpoint_every <= self.epochs
            ), "Checkpoint frequency greater than number of epochs. Current program won't checkpoint models."
        # Attribute set up within init function.
        if self.checkpoint_dir_path:
            self.experiment_dir = os.path.dirname(self.checkpoint_dir_path)
        else:
            self.experiment_dir = os.getcwd()

    def train(self):
        """
        Train UNet in conjuction with DDPM.
        """
        print("Training DDPM on device:", self.device)
        train_metrics = {"train_diff_loss": []}
        self.dataloader = DataLoader(
            self.dataset, self.batch_size, shuffle=True, pin_memory=True
        )
        for epoch in range(self.epochs):
            print("\nEpoch %d\n" % (epoch + 1))
            for idx, (mbatch_x, mbatch_y) in enumerate(self.dataloader):
                mbatch_x = mbatch_x.to(self.device)
                mbatch_y = mbatch_y.to(self.device)
                self.optimizer.zero_grad()
                self.loss = self.ddpm.l_simple(mbatch_x)
                if idx % 100 == 0:
                    print("Diffusion Loss %.3f" % self.loss)
                self.loss.backward()
                self.optimizer.step()
            self.sample(f"epoch_{epoch}")
            train_metrics["train_diff_loss"].append(self.loss.item())
            if self.checkpoint_every and (epoch % self.checkpoint_every == 0):
                checkpoint_name = "ddpm_checkpoint_e_%d_loss_%.3f.pt" % (
                    (epoch + 1),
                    self.loss,
                )
                checkpoint_path = os.path.join(
                    self.checkpoint_dir_path, checkpoint_name
                )
                checkpoint(
                    checkpoint_path,
                    epoch + 1,
                    self.noise_predictor.state_dict(),
                    self.optimizer.state_dict(),
                    self.loss,
                )
        if self.checkpoint_every:
            checkpoint_name = "ddpm_fully_trained_e_%d_loss_%.3f.pt" % (
                self.epochs,
                self.loss,
            )
            checkpoint_path = os.path.join(self.checkpoint_dir_path, checkpoint_name)
            checkpoint(
                checkpoint_path,
                self.epochs,
                self.noise_predictor.state_dict(),
                self.optimizer.state_dict(),
                self.loss,
            )
        return train_metrics

    @torch.no_grad()
    def sample(self, title: str = ""):
        """
        Sample from diffusion model
        """
        if isinstance(self.dataset, ModelsDataset):
            self.sample_dimensions = self.dataset.tensor_sample_dim[1:]
        x_T = torch.randn(
            [
                self.num_gen_samples,
                self.sample_channels,
                self.sample_dimensions[0],
                self.sample_dimensions[1],
            ]
        )  # Sample from Standard Gaussian (distribution at end of diffusion process) in the dimensions of original sample and sample according to the number of samples to generate.
        x_t = x_T
        for t in range(self.diffusion_steps):
            t_ = self.diffusion_steps - t - 1
            x_t = self.ddpm.sample_p_t_reverse_process(
                x_t, x_t.new_full((self.num_gen_samples,), t_, dtype=torch.long)
            )
        # sample1, sample2, sample3, sample4, sample5 = torch.chunk(x_t, 5, 0)
        x_t = x_t.cpu().numpy()
        restored_samples = []
        for i in range(self.num_gen_samples):
            sample = x_t[i]
            if isinstance(self.dataset, ModelsDataset):
                restored_samples.append(self.dataset.restore_original_tensor(sample))
            else:
                plt.imsave(
                    f"{self.experiment_dir}/{title}_gen_sample_{i}.png",
                    np.squeeze(sample),
                )
        return restored_samples

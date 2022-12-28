import torch, sys
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

sys.path.append("../")
from models.ddpm import DDPM
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
        sample_dimensions: tuple[int],
        channel_multipliers: list[int],
        is_attention: list[bool],
        num_gen_samples: int,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        dataset: Dataset,
        device: str,
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
            sample_channels, num_channels, channel_multipliers, is_attention
        )
        self.ddpm = DDPM(self.noise_predictor, self.diffusion_steps, self.device)

        self.optimizer = Adam(self.noise_predictor.parameters(), self.learning_rate)

    def train(self):
        """
        Train UNet in conjuction with DDPM.
        """
        self.dataloader = DataLoader(
            self.dataset, self.batch_size, shuffle=True, pin_memory=True
        )
        for epoch in range(self.epochs):
            for idx, (mbatch_x, mbatch_y) in enumerate(self.dataloader):
                mbatch_x = mbatch_x.to(self.device)
                mbatch_y = mbatch_y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.ddpm.l_simple(mbatch_x)
                if idx % 100 == 0:
                    print("Diffusion Loss %.3f" % loss)
                loss.backward()
                self.optimizer.step()
            self.sample()

    @torch.no_grad()
    def sample(self):
        """
        Sample from diffusion model
        """
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
            if t > 1:
                x_t = self.ddpm.sample_p_t_reverse_process(
                    x_t, x_t.new_full((self.num_gen_samples,), t, dtype=torch.long)
                )
        plt.imshow(x_t)

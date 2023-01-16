# Script based on https://nn.labml.ai/diffusion/ddpm/evaluate.html; Full-credits to them.
import numpy as np
import torch, os
import matplotlib.pyplot as plt
from frameworks.ddpm_template import DDPMDiffusion
from models.ddpm import gather


class DDPMSampler:
    def __init__(
        self,
        diffusion_process: DDPMDiffusion,
        sample_channels: int,
        sample_size: tuple,
        device: str,
        checkpoint_dir: str,
    ):
        self.device = device
        self.diffusion_process = diffusion_process
        self.sample_channels = sample_channels
        self.sample_size = sample_size
        self.checkpoint_dir = checkpoint_dir
        self.experiment_dir = os.path.dirname(self.checkpoint_dir)

        self.n_steps = diffusion_process.diffusion_steps  # number of diffusion steps
        self.noise_predictor = diffusion_process.noise_predictor  # denoising model
        self.beta = diffusion_process.ddpm.betas
        self.alpha = diffusion_process.ddpm.alphas
        self.alpha_bar = diffusion_process.ddpm.alpha_bar
        alphabar_t_minus_1 = torch.cat(
            [self.alpha_bar.new_ones((1,)), self.alpha_bar[:-1]]
        )  # tensor filled with ones and with shape defined by layout

        self.beta_tilde = self.beta * (1 - alphabar_t_minus_1) / (1 - self.alpha_bar)
        self.mu_tilde_coef_1 = (
            self.beta * (alphabar_t_minus_1**0.5) / (1 - self.alpha_bar)
        )

        self.mu_tilde_coef_2 = (
            (self.alpha**0.5) * (1 - alphabar_t_minus_1) / (1 - self.alpha_bar)
        )
        self.sigma2 = self.beta

    def show_image(self, img, title: str = ""):
        img = img.clip(
            0, 1
        )  # forces values to be between 0 and 1. If value lower than 0 -> 0 if value greater than 1 -> 1.
        img = (
            img.cpu().detach().numpy().transpose(1, 2, 0)
        )  # Changes view so that channels are last dim.
        # plt.imshow(
        #     img.transpose(1, 2, 0)
        # )
        # plt.title(title)
        # plt.show()
        plt.imsave(f"{self.experiment_dir}/{title}.jpg", np.squeeze(img))

    def sample(self, n_samples: int = 16):
        xt = torch.randn(
            [n_samples, self.sample_channels, self.sample_size[0], self.sample_size[1]],
            device=self.device,
        )
        x0 = self._sample_x0(xt, self.n_steps)
        for i in range(n_samples):
            self.show_image(x0[i], f"generated_image_{i}")

    def _sample_x0(self, xt: torch.Tensor, n_steps: int):
        n_samples = xt.shape[0]
        for t_ in range(n_steps):
            t = n_steps - t_ - 1
            xt = self.diffusion_process.ddpm.sample_p_t_reverse_process(
                xt, xt.new_full((n_samples,), t, dtype=torch.long
            ))  # sample from reverse processs
            return xt

    def p_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.tensor):
        """
        Estimation of x_0 (original sample)
        """
        alpha_bar = torch.gather(self.alpha_bar, 0, t)
        return (xt - (1 - alpha_bar) ** 0.5 * eps) / (alpha_bar**0.5)

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, eps_theta: torch.Tensor):
        """
        Sample from p_theta(x(t-1) | x(t))
        """
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coefficient = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha**0.5) * (xt - eps_coefficient * eps_theta)
        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + var(**0.5) * eps

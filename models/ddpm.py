### Script based on: https://nn.labml.ai/diffusion/ddpm/index.html; Full credits to them.

import torch
from typing import Tuple, Optional
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape (taken from LabML utils script)"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


class DDPM:
    def __init__(
        self,
        eps_nn: nn.Module,
        diffusion_steps: int,
        device: str,
    ):
        super().__init__()
        self.eps_nn = eps_nn.to(device)
        self.betas = torch.linspace(0.0001, 0.02, diffusion_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(
            self.alphas, dim=0
        )  # creates tensor of same dimension as self.alphas but with each element being the result of the cumulative product up to this point, i.e. it looks like: [1, 1*0.999, 1*0.999*0.998, 1*0.999*0.998*0.997,...]
        self.diffusion_steps = diffusion_steps
        self.covariance_reverse_process = (
            self.betas
        )  # recall \sigma_\theta (x_T, t) is fixed in DDPM and learned in improved DDPM
        self.device = device

        self.name = self.__class__.__name__

    def q_xt_given_x0(self, x0: torch.Tensor, t: torch.Tensor):
        """
        This method defines the parameters of q(x_t | x_0) and is expressed
        as a gaussian. This method returns the mean and covariance of the q(x_t | x_0)
        distribution at time t.

        Args:
            - x0: original sample from forward diffusion takes place
            - t: number of t'th step in diffusion process.
        """
        # alpha_bar_t = self.compute_alpha_bar(t)
        # mean = torch.einsum("abcd,a->abcd", x0, alpha_bar_t ** (0.5)).to(self.device)
        # covariance = 1 - alpha_bar_t
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        # return mean.to(self.device), covariance.to(self.device)
        return mean.to(self.device), var.to(self.device)

    def sample_q_xt_given_x0(
        self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor = None
    ):
        """
        Returns samples from q(x_t | x_0) as part of forward diffusion
        process.
        """
        if eps is None:
            # eps = torch.distributions.Normal(0, 1).sample(x0.size())
            eps = torch.randn_like(x0).to(self.device)
        # mean, cov = self.q_xt_given_x0(x0, t)
        mean, var = self.q_xt_given_x0(x0, t)
        # mean = mean.to(self.device)
        # cov = cov.to(self.device)

        eps = eps.to(self.device)
        # return mean + torch.einsum("abcd, a->abcd", eps, (cov ** (1 / 2)))
        return mean + (var**0.5) * eps

    def sample_p_t_reverse_process(self, xt: torch.Tensor, t: torch.Tensor):
        """
        This method returns a sample from the reverse distribution, p_\theta(x_t-1 | x_t).
        This distribution is assumed to be a Gaussian because of the apparent similarity in
        functional form between the forward and reverse processes when the diffusion step noise is marginal.

        """
        xt = xt.to(self.device)
        t = t.to(self.device)
        eps_theta = self.eps_nn(
            xt, t
        )  # Get predicted noise from reverse process from model that estimates it, \epsilon_\theta(x_t, t), which takes current diffusion sample, x_t and diffusion step t as inputs
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alphas, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha**0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.covariance_reverse_process, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var**0.5) * eps
        # alpha_t = self.alphas[t]
        # alpha_t = torch.gather(self.alphas, 0, t)
        # alpha_bar_t = self.compute_alpha_bar(t)
        # eps_theta_factor = (1 - alpha_t) / (1 - alpha_bar_t) ** (1 / 2)
        # mean_p_t = torch.einsum(
        #     "abcd, a->abcd",
        #     (xt - torch.einsum("abcd, a->abcd", eps_theta, eps_theta_factor)),
        #     1 / alpha_t ** (1 / 2),
        # )
        # var_p_t = torch.gather(self.covariance_reverse_process, 0, t)
        # eps = torch.distributions.Normal(0, 1).sample(xt.size()).to(self.device)
        # return mean_p_t + torch.einsum("abcd, a->abcd", eps, (var_p_t) ** (1 / 2))

    def l_simple(self, x0: torch.Tensor, noise: torch.Tensor = None):
        """
        Simplified loss function for the training of the network estimating
        the parameters of Gaussian distribution estimating the reverse diffusion process.
        """
        batch_size = x0.size()[0]
        # Sample a random t-step for each sample in the batch
        t = torch.randint(
            0, self.diffusion_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if not noise:
            # noise = torch.distributions.Normal(0, 1).sample(x0.size())
            noise = torch.randn_like(x0).to(self.device)
        # noise = noise.to(self.device)
        # xt = self.sample_q_xt_given_x0(x0, t, noise)
        xt = self.sample_q_xt_given_x0(x0, t, eps=noise)
        eps_theta = self.eps_nn(xt, t)
        return F.mse_loss(noise, eps_theta)

    def compute_alpha_bar(self, t: torch.Tensor):
        alphas = torch.gather(self.alpha_bar, 0, t)
        return alphas

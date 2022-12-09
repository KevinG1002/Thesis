import torch
from torch.distributions.laplace import Laplace
from .param_distribution import ParameterDistribution


class LaPlaceDistribution(ParameterDistribution):
    def __init__(self, mu: torch.Tensor, rho: torch.Tensor):
        super(LaPlaceDistribution, self).__init__()
        self.mu = mu
        self.rho = rho

    @property
    def scale_factor(self):
        return torch.log(1 + torch.exp(torch.Tensor([self.rho])))

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        return Laplace(self.mu, self.scale_factor).log_prob(values)

    def sample(self) -> torch.Tensor:
        return Laplace(self.mu, self.scale_factor).sample(self.mu.size())

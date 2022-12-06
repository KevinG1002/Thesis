import torch
from torch.distributions.laplace import Laplace
from .param_distribution import ParameterDistribution


class LaPlaceDistribution(ParameterDistribution):
    def __init__(self, mu: torch.Tensor, var: torch.Tensor):
        super(LaPlaceDistribution, self).__init__()
        self.mu = mu
        self.var = var

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        return Laplace(self.mu, self.var).log_prob(values)

    def sample(self) -> torch.Tensor:
        return Laplace(self.mu, self.var).sample(self.mu.size())

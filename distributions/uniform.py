import torch
from torch.distributions.uniform import Uniform

from .param_distribution import ParameterDistribution


class UniformDistribution(ParameterDistribution):
    def __init__(self, low: float, high: float):
        super(UniformDistribution, self).__init__()
        self.low = low
        self.high = high

    def log_likelihood(self, values) -> torch.Tensor:
        return Uniform(self.low, self.high).log_prob(values)

    def sample(self) -> torch.Tensor:
        return Uniform(self.low, self.high).sample()

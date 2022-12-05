import torch
from torch.nn import Module
from ..distributions.param_distribution import ParameterDistribution


class BayesianLayer(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        prior: ParameterDistribution,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.prior = prior

    def forward(self, x: torch.Tensor):
        pass

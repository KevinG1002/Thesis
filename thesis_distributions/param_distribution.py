import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class ParameterDistribution(nn.Module, ABC):
    """
    Class taken from Andreas Krause' PAI BNN assignment in 2021.
    """

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        """
        Returns torch tensor of log likelihood of values passed as input.
        """
        ...

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Returns sample from parameter distribution.
        """
        ...

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """
        Necessary for nn.Module super class. Just returns log-likelihood of vector of values.
        """
        return self.log_likelihood(values)

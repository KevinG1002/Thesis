import torch
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from .param_distribution import ParameterDistribution


class UnivariateGaussian(ParameterDistribution):
    def __init__(self, mu: float, var: float):
        super(UnivariateGaussian, self).__init__()
        self.mu = mu
        self.var = var

    def log_likelihood(self, values) -> torch.Tensor:
        NLL = (
            (1 / 2) * torch.log(2 * torch.pi)
            + torch.log(self.var ** (1 / 2))
            + ((values - self.mu) ** 2 / (2 * self.var))
        )
        return -NLL

    def sample(self) -> torch.Tensor:
        return Normal(self.mu, self.var).sample()


class MultivariateDiagonalGaussian(ParameterDistribution):
    def __init__(self, mu: torch.Tensor, var: torch.Tensor):
        super(MultivariateDiagonalGaussian, self).__init__()
        self.mu = mu
        self.var = var

    def log_likelihood(self, values) -> torch.Tensor:
        """
        Recall, log likelihood of multivariate gaussian given by:
        LL = -k/2 ln(2π) - ln(det(\Sigma)^(1/2)) - 1/2 (x - mu)^T \Sigma^(-1) (x-mu)
        """
        k = values.size()[0]
        NLL = (
            (k / 2) * torch.log(2 * torch.pi)
            + torch.log(torch.det(self.var) ** (1 / 2))
            + (1 / 2)
            * ((values - self.mu).T @ torch.inverse(self.var) @ (values - self.mu))
        )
        return -NLL

    def sample(self) -> torch.Tensor:
        return Normal(self.mu, self.var).sample(self.mu.size())


class ScaleMixtureGaussian(ParameterDistribution):
    def __init__(self, p: float, mu: float, var: float):
        super(ScaleMixtureGaussian, self).__init__()
        self.p = p
        self.mu = mu
        self.var = var

    def log_likelihood(self, values):
        ...

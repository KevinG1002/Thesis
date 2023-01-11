import torch
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from .param_distribution import ParameterDistribution

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class UnivariateGaussian(ParameterDistribution):
    def __init__(self, mu: float, rho: float):
        super(UnivariateGaussian, self).__init__()
        self.mu = mu
        self.rho = rho
        self.name = self._get_name()

    @property
    def sigma(self):
        return torch.log(1 + torch.exp(torch.Tensor([self.rho]))).to(DEVICE)

    def log_likelihood(self, values) -> torch.Tensor:
        NLL = (
            (1 / 2) * torch.log(torch.Tensor([2 * torch.pi]).to(DEVICE))
            + torch.log(torch.Tensor([self.sigma]).to(DEVICE))
            + ((values - self.mu) ** 2 / (self.sigma**2))
        )
        return -NLL

    def sample(self) -> torch.Tensor:
        return Normal(self.mu, self.sigma).sample()


class MultivariateDiagonalGaussian(ParameterDistribution):
    def __init__(self, mu: torch.Tensor, rho: torch.Tensor):
        super(MultivariateDiagonalGaussian, self).__init__()
        self.mu = mu
        self.rho = rho

    @property
    def sigma(self):
        return torch.log(1 + torch.exp(self.rho)).to(DEVICE)

    def log_likelihood(self, values) -> torch.Tensor:
        """
        Recall, log likelihood of multivariate gaussian given by:
        LL = -k/2 ln(2π) - ln(det(\Sigma)^(1/2)) - 1/2 (x - mu)^T \Sigma^(-1) (x-mu)
        Here, values, mu and var all have the same dimension. Therefore the log-likelihood
        is computed in an element wise fashion
        """
        NLL = (
            (1 / 2) * torch.log(torch.Tensor([2 * torch.pi]).to(DEVICE))
            + torch.log(self.sigma)
            + ((values - self.mu) ** 2 / (2 * self.sigma**2))
        )
        return -NLL

    def sample(self) -> torch.Tensor:
        return Normal(self.mu, self.sigma).sample(self.mu.size())


class ScaleMixtureGaussian(ParameterDistribution):
    def __init__(
        self,
        p: torch.Tensor,
        mu_set: torch.Tensor,
        rho_set: torch.Tensor,
    ):
        super(ScaleMixtureGaussian, self).__init__()
        assert (
            p.size() == self.mu_set.size()[0]
        ), "mixture weights not equal to number of component distributions"
        self.p = p
        self.mu_set = mu_set
        self.rho_set = rho_set
        mix = Categorical(self.p)
        component_dists = Normal(self.mu_set, self.sigma_set)
        self.scale_mix = MixtureSameFamily(mix, component_dists)

    @property
    def sigma_set(self):
        return torch.log(1 + torch.exp(self.rho_set))

    def log_likelihood(self, values) -> torch.Tensor:
        return self.scale_mix.log_prob(values)

    def sample(self) -> torch.Tensor:
        return self.scale_mix.sample(self.mu_set.size())

import torch
from typing import Type
from torch.nn import Module, Parameter
import torch.nn.functional as F
import sys

sys.path.append("..")
from thesis_distributions.param_distribution import ParameterDistribution
from thesis_distributions.gaussians import (
    UnivariateGaussian,
    MultivariateDiagonalGaussian,
)


class BayesianLayer(Module):
    """
    Implementation of a bayesian layer that will be used for the Bayesian Neural Network (fully connected).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        prior: ParameterDistribution,
    ):
        """
        Initialisation of the Bayesian Layer.
        Args:
            - in_features (int): denotes the dimension of the input to the current layer
            - out_features (int): denotes the dimension of the output of the current layer
            - bias (bool): denotes whether a bias term is used for this layer or not.
            - weight_prior (Type[ParameterDistribution]): prior distributon posed over the weights of the network.
            - bias_prior (Type[ParameterDistribution]): prior distributon posed over the biases in the network.
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.in_features = in_features
        self.out_features = out_features
        self.prior = prior.to(self.device)
        self.bias = bias

        if bias:
            self.bias_prior = (
                prior if prior else UnivariateGaussian(torch.zeros(1), torch.ones(1))
            )

            assert isinstance(self.bias_prior, ParameterDistribution)
            assert not any(
                True for _ in self.bias_prior.parameters()
            ), "Prior cannot have parameters"

            self.bias_var_posterior = MultivariateDiagonalGaussian(
                mu=Parameter(torch.Tensor(self.out_features).normal_(0, 0.1)),
                rho=Parameter(torch.Tensor(self.out_features).normal_(0, 1)),
            ).to(self.device)
            assert isinstance(self.bias_var_posterior, ParameterDistribution)
            assert any(True for _ in self.bias_var_posterior.parameters())
        else:
            self.bias_var_posterior = None
        self.weight_prior = (
            prior
            if prior
            else UnivariateGaussian(torch.zeros(1), torch.ones(1)).to(self.device)
        )
        assert isinstance(self.weight_prior, ParameterDistribution)
        assert not any(
            True for _ in self.weight_prior.parameters()
        ), "Prior cannot have parameters"
        # estimate of weight variational posterior, start close to zero for numerical stability; these will be optimized with time.
        self.weights_var_posterior = MultivariateDiagonalGaussian(
            mu=Parameter(
                torch.Tensor(self.out_features, self.in_features).normal_(0, 1)
            ),
            rho=Parameter(
                torch.Tensor(self.out_features, self.in_features).normal_(0, 1)
            ),
        ).to(self.device)
        assert isinstance(self.weights_var_posterior, ParameterDistribution)

    def forward(self, x: torch.Tensor):
        """
        Forward pass across the layer
        Args:
            x(torch.Tensor): input to bayesian layer.
        """

        epsilon = (
            torch.distributions.Normal(0, 1)
            .sample((self.out_features, self.in_features))
            .to(self.device)
        )

        weights = self.weights_var_posterior.mu + (
            self.weights_var_posterior.sigma * epsilon
        )
        log_prior = self.weight_prior.log_likelihood(weights).sum()
        log_variational_posterior = self.weights_var_posterior.log_likelihood(
            weights
        ).sum()

        if self.bias:
            eps = (
                torch.distributions.Normal(0, 1)
                .sample([self.out_features])
                .to(self.device)
            )
            bias = self.bias_var_posterior.mu + eps * (self.bias_var_posterior.sigma)

            log_prior += self.bias_prior.log_likelihood(bias).sum()
            log_variational_posterior += self.bias_var_posterior.log_likelihood(
                bias
            ).sum()

            return F.linear(x, weights, bias), log_prior, log_variational_posterior

        else:
            return F.linear(x, weights, bias=None), log_prior, log_variational_posterior

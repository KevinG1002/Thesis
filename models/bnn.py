import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from frameworks.bayesian_layer import BayesianLayer
from distributions.param_distribution import ParameterDistribution


class SimpleBNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        number_of_classes: int,
        prior_dist: ParameterDistribution,
        bias: bool = True,
    ):
        super(SimpleBNN, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.in_features = in_features
        self.number_of_classes = number_of_classes
        self.prior_dist = prior_dist.to(self.device)
        self.name = self._get_name()

        ## Architecture Structure ##
        self.bl_1 = BayesianLayer(
            in_features=self.in_features,
            out_features=400,
            bias=bias,
            prior=self.prior_dist,
        )
        self.bl_2 = BayesianLayer(
            in_features=400, out_features=600, bias=bias, prior=self.prior_dist
        )

        self.bl_3 = BayesianLayer(
            in_features=600,
            out_features=number_of_classes,
            bias=bias,
            prior=self.prior_dist,
        )

    def forward(self, x):
        tot_log_prior = torch.tensor(0.0).to(self.device)
        tot_log_var_posterior = torch.tensor(0.0).to(self.device)

        x, log_prior, log_var_posterior = self.bl_1(x)
        tot_log_prior += log_prior
        tot_log_var_posterior += log_var_posterior
        x = F.relu(x)
        x, log_prior, log_var_posterior = self.bl_2(x)
        tot_log_prior += log_prior
        tot_log_var_posterior += log_var_posterior
        x = F.relu(x)
        x, log_prior, log_var_posterior = self.bl_3(x)
        tot_log_prior += log_prior
        tot_log_var_posterior += log_var_posterior
        x = F.relu(x)
        return F.softmax(x, dim=1), tot_log_prior, tot_log_var_posterior

    def predict(self, x: torch.Tensor, num_mc_samples: int):
        probability_samples = torch.stack(
            [self.forward(x)[0] for _ in range(num_mc_samples)]
        )
        estimated_probability = torch.mean(probability_samples, dim=0)
        assert estimated_probability.shape == (x.shape[0], self.number_of_classes)
        assert torch.allclose(
            torch.sum(estimated_probability, dim=1), torch.tensor(1.0)
        )

        return estimated_probability

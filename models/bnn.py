import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("..")
from frameworks.bayesian_layer import BayesianLayer
from distributions.param_distribution import ParameterDistribution


class SimpleBNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        number_of_classes: int,
        prior_dist: ParameterDistribution,
    ):
        super(SimpleBNN, self).__init__()
        self.in_features = in_features
        self.number_of_classes = number_of_classes
        self.prior_dist = prior_dist

        ## Architecture Structure ##
        self.bl_1 = BayesianLayer(
            in_features=self.in_features,
            out_features=400,
            bias=True,
            prior=self.prior_dist,
        )
        self.bl_2 = BayesianLayer(
            in_features=400, out_features=400, bias=True, prior=self.prior_dist
        )
        self.bl_3 = BayesianLayer(
            in_features=400, out_features=100, bias=True, prior=self.prior_dist
        )
        self.bl_4 = BayesianLayer(
            in_features=100,
            out_features=number_of_classes,
            bias=True,
            prior=self.prior_dist,
        )

    def forward(self, x):
        x = F.relu(self.bl_1(x))
        x = F.relu(self.bl_2(x))
        x = F.relu(self.bl_3(x))
        x = F.relu(self.bl_4(x))
        return torch.sof

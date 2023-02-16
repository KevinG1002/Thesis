import pyro
import torch
from typing import Callable
import numpy as np
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.distributions.transforms import NeuralAutoregressive, AffineCoupling
from pyro.infer import SVI, Trace_ELBO, MCMC, TraceGraph_ELBO, Predictive
from pyro.infer.autoguide import (
    AutoNormalizingFlow,
    AutoNormal,
    AutoIAFNormal,
    AutoMultivariateNormal,
)
import torch.nn as nn
from pyro import poutine
from sklearn.manifold import TSNE
import pyro.optim
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from models.mlp import MLP
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import graphviz
import pandas as pd
from models.mlp import MLP


class MLP_PGM(object):
    def __init__(
        self,
        learning_rate: float,
        batch_size: int,
        base_model: nn.Module,
        dataset_len: int,
        guide: Callable = None,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.base_model = base_model
        self.dataset_len = dataset_len
        self.layer_structure = (
            param.size()[0] for param in nn.parameters() if len(param.size()) == 1
        )
        self.weight_matrix_dims = [
            (self.layer_structure[i], self.layer_structure[i + 1])
            for i in range(len(self.layer_structure) - 1)
        ]
        self.bias_dims = [
            (self.layer_structure[k], 1) for k in range(1, len(self.layer_structure))
        ]
        self.guide: Callable = guide if guide else self.pgm_guide()

        self.trace = poutine.trace(self.pgm_model).get_trace()

    def pgm_model(
        self,
        layerwise_weight_dataset: list[torch.Tensor] = None,
        layerwise_bias_dataset: list[torch.Tensor] = None,
    ):
        self.subsample_size = self.batch_size if layerwise_weight_dataset else None
        x = torch.randn(
            self.subsample_size if self.subsample_size else self.dataset_len, 784, 1
        )

        layer_idx = 0
        for _ in range(
            len(self.weight_matrix_dims)
        ):  # iterate over number of weight matrices
            upper_bound = (1 / self.layer_structure[layer_idx]) ** (1 / 2) * torch.ones(
                self.layer_structure[layer_idx]
            )
            lower_bound = (
                -upper_bound
            )  # define lower and upper bound for uniform distributions used to initialise weights for NNs
            # print(upper_bound[0])
            b_upper_bound = (
                1 / self.layer_structure[layer_idx + 1] ** (1 / 2)
            ) * torch.ones(self.layer_structure[layer_idx + 1])
            b_lower_bound = -b_upper_bound
            # print(b_lower_bound[0])

            with pyro.plate(
                f"Layer_{layer_idx+1}",
                self.dataset_len,
                self.subsample_size,
                dim=-3,
                use_cuda=True,
            ) as ind:
                print(ind.device)
                ind = ind.cpu()
                # print(layer_idx)
                b = pyro.sample(
                    f"b_{layer_idx+1}",
                    dist.Uniform(b_lower_bound, b_upper_bound),
                    obs=layerwise_bias_dataset[layer_idx][ind]
                    if layerwise_bias_dataset
                    else None,
                )  # Sample b vector from uniform
                # w = torch.empty(self.weight_matrix_dims[layer_idx]) # create placeholder tensor for weights so that we can concatenate independent variational parameters to do weight multiplication
                # for j in range(self.weight_matrix_dims[layer_idx][1])
                with pyro.plate(
                    f"Layer_Weights_{layer_idx+1}",
                    self.weight_matrix_dims[layer_idx][1],
                    dim=-2,
                    use_cuda=True,
                ):
                    # print(weight_layer_dataset[layer_idx].size())
                    d = dist.Uniform(lower_bound, upper_bound)
                    w = pyro.sample(
                        f"w_{layer_idx+1}",
                        d,
                        obs=layerwise_weight_dataset[layer_idx][ind.cpu()].permute(
                            0, 2, 1
                        )
                        if layerwise_weight_dataset
                        else None,
                    )

                    if layerwise_weight_dataset:
                        print(
                            "Layer %d, Bias %s, Weight %s, Input %s"
                            % (layer_idx + 1, b.size(), w.size(), x.size())
                        )
                        act = torch.relu(
                            torch.matmul(w, x).squeeze() + b
                        )  # .unsqueeze(-1))
                    else:
                        b = b.permute(0, 2, 1).squeeze(-1)
                        print(
                            "Layer %d, Bias %s, Weight %s, Input %s"
                            % (layer_idx + 1, b.size(), w.size(), x.size())
                        )
                        # print(torch.matmul(x.squeeze(), w.mT).size())
                        act = torch.relu(torch.matmul(w, x).squeeze() + b)  # .squeeze()
                        # print("Weight_size", w.size())

                    print(
                        "Layer %d, Bias %s, Weight %s, Act %s"
                        % (layer_idx + 1, b.size(), w.size(), act.size())
                    )

                    # activations.append(act)

                    # cov = torch.eye(act.size(1))

            with pyro.plate(f"Activations_{layer_idx+1}", 1, dim=-3, use_cuda=True):
                cov = torch.stack(
                    [
                        torch.eye(self.bias_dims[layer_idx][0])
                        for _ in range(act.size()[0])
                    ],
                    dim=0,
                )
                h_dist = dist.MultivariateNormal(act.squeeze(), cov)
                # print("Test Sample size", h_dist.sample().size())
                h = pyro.sample(f"h_{layer_idx +1}", h_dist).squeeze()
                # print("Latent Sample", h.size())
                x = h.unsqueeze(-1)
                layer_idx += 1

    def pgm_guide(self):
        pass

    def svi_train():
        pass

    def sample_latents(self, num_samples):
        with pyro.plate("samples", 1000, dim=-4):
            samples = self.guide(None, None)
        return samples

    @property
    def latents(self):
        names = []
        for name, _ in pyro.get_param_store().items():
            names.append(name)
            return names

    @property
    def observed(self):
        names = []


def mlp_pgm_final(
    weight_layer_dataset: list[torch.Tensor] = None,
    bias_layer_dataset: list[torch.Tensor] = None,
):
    dataset_len = weight_layer_dataset[0].size()[0] if weight_layer_dataset else 1
    subsample_size = 2 if weight_layer_dataset else None
    layer_structure = (784, 128, 64, 10)
    weight_matrix_dims = [
        (layer_structure[i], layer_structure[i + 1])
        for i in range(len(layer_structure) - 1)
    ]
    bias_dims = [(layer_structure[k], 1) for k in range(1, len(layer_structure))]
    # print(weight_matrix_dims)
    x = torch.randn(subsample_size if subsample_size else dataset_len, 784, 1)
    layer_idx = 0
    for _ in range(len(weight_matrix_dims)):  # iterate over number of weight matrices
        upper_bound = (1 / layer_structure[layer_idx]) ** (1 / 2) * torch.ones(
            layer_structure[layer_idx]
        )
        lower_bound = (
            -upper_bound
        )  # define lower and upper bound for uniform distributions used to initialise weights for NNs
        # print(upper_bound[0])
        b_upper_bound = (1 / layer_structure[layer_idx + 1] ** (1 / 2)) * torch.ones(
            layer_structure[layer_idx + 1]
        )
        b_lower_bound = -b_upper_bound
        # print(b_lower_bound[0])

        with pyro.plate(
            f"Layer_{layer_idx+1}", dataset_len, subsample_size, dim=-3, use_cuda=True
        ) as ind:
            print(ind.device)
            ind = ind.cpu()
            # print(layer_idx)
            b = pyro.sample(
                f"b_{layer_idx+1}",
                dist.Uniform(b_lower_bound, b_upper_bound),
                obs=bias_layer_dataset[layer_idx][ind] if bias_layer_dataset else None,
            )  # Sample b vector from uniform
            # w = torch.empty(weight_matrix_dims[layer_idx]) # create placeholder tensor for weights so that we can concatenate independent variational parameters to do weight multiplication
            # for j in range(weight_matrix_dims[layer_idx][1])
            with pyro.plate(
                f"Layer_Weights_{layer_idx+1}",
                weight_matrix_dims[layer_idx][1],
                dim=-2,
                use_cuda=True,
            ):
                # print(weight_layer_dataset[layer_idx].size())
                d = dist.Uniform(lower_bound, upper_bound)
                w = pyro.sample(
                    f"w_{layer_idx+1}",
                    d,
                    obs=weight_layer_dataset[layer_idx][ind.cpu()].permute(0, 2, 1)
                    if weight_layer_dataset
                    else None,
                )

                if weight_layer_dataset:
                    print(
                        "Layer %d, Bias %s, Weight %s, Input %s"
                        % (layer_idx + 1, b.size(), w.size(), x.size())
                    )
                    act = torch.relu(
                        torch.matmul(w, x).squeeze() + b
                    )  # .unsqueeze(-1))
                else:
                    b = b.permute(0, 2, 1).squeeze(-1)
                    print(
                        "Layer %d, Bias %s, Weight %s, Input %s"
                        % (layer_idx + 1, b.size(), w.size(), x.size())
                    )
                    # print(torch.matmul(x.squeeze(), w.mT).size())
                    act = torch.relu(torch.matmul(w, x).squeeze() + b)  # .squeeze()
                    # print("Weight_size", w.size())

                print(
                    "Layer %d, Bias %s, Weight %s, Act %s"
                    % (layer_idx + 1, b.size(), w.size(), act.size())
                )

                # activations.append(act)

                # cov = torch.eye(act.size(1))

        with pyro.plate(f"Activations_{layer_idx+1}", 1, dim=-3, use_cuda=True):
            cov = torch.stack(
                [torch.eye(bias_dims[layer_idx][0]) for _ in range(act.size()[0])],
                dim=0,
            )
            h_dist = dist.MultivariateNormal(act.squeeze(), cov)
            # print("Test Sample size", h_dist.sample().size())
            h = pyro.sample(f"h_{layer_idx +1}", h_dist).squeeze()
            # print("Latent Sample", h.size())
            x = h.unsqueeze(-1)
            layer_idx += 1


def split_weights_biases(nn: torch.nn.Module):
    return [param.mT for param in nn.parameters() if len(param.size()) > 1], [
        param for param in nn.parameters() if len(param.size()) < 2
    ]


def tensor_dataset_layer_wise(
    num_layers_in_nn: int, nn_layers: list[list[torch.Tensor]]
):
    layer_datasets = []
    for i in range(num_layers_in_nn):
        layer_set = []
        for j in range(len(nn_layers)):
            layer_set.append(nn_layers[j][i])
        layer_datasets.append(torch.stack(layer_set, 0))
    return layer_datasets


def layer_wise_dataset(nn_dataset: list[list]):
    weight_dataset, bias_dataset = [], []
    for i in range(len(nn_dataset)):
        weights, biases = split_weights_biases(nn_dataset[i])
        weight_dataset.append(weights)
        bias_dataset.append(biases)

    layerwise_weight_datasets = tensor_dataset_layer_wise(len(weights), weight_dataset)
    layerwise_bias_datasets = tensor_dataset_layer_wise(len(weights), bias_dataset)
    return layerwise_weight_datasets, layerwise_bias_datasets


def test():
    nn_list = [MLP() for _ in range(5)]
    layerwise_weight_datasets, layerwise_bias_datasets = layer_wise_dataset(nn_list)
    print([x.size() for x in layerwise_weight_datasets])
    print([x.size() for x in layerwise_bias_datasets])
    pyro.render_model(mlp_pgm_final, model_args=(None, None), render_params=True)


def run_inference():
    pyro.clear_param_store()
    nn_list = [MLP() for _ in range(5)]
    layerwise_weight_datasets, layerwise_bias_datasets = layer_wise_dataset(nn_list)
    print([x.size() for x in layerwise_weight_datasets])
    print([x.size() for x in layerwise_bias_datasets])

    # trace = poutine.trace(mlp_pgm_final).get_trace()
    # trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
    # print(trace.format_shapes())
    # exit(0)
    mlp_pgm_guide = AutoMultivariateNormal(mlp_pgm_final)
    adam = pyro.optim.Adam({"lr": 0.02})  # Consider decreasing learning rate.
    elbo = Trace_ELBO()
    svi = pyro.infer.SVI(mlp_pgm_final, mlp_pgm_guide, adam, elbo)

    losses = []
    for step in range(2000):  # Consider running for more steps.
        loss = svi.step(layerwise_weight_datasets, layerwise_bias_datasets)
        losses.append(loss)
        print(loss)
        if step % 10 == 0:
            print("Elbo loss: {}".format(loss))

    plt.figure(figsize=(5, 2))
    plt.plot(losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")


if __name__ == "__main__":
    test()
    # run_inference()

import torch
import torch.nn as nn


def split_weights_biases(nn: nn.Module):
    """
    Function that returns tuple of lists torch tensors. First element
    in tuple contains the weight tensors. The second element contains
    the bias tensors.
    """
    return [param.mT for param in nn.parameters() if len(param.size()) > 1], [
        param for param in nn.parameters() if len(param.size()) < 2
    ]


def tensor_dataset_layer_wise(
    num_layers_in_nn: int, nn_layers: list[list[torch.Tensor]]
):
    """
    Returns a list of tensors, each representing a layerwise dataset.
    """
    layer_datasets = []
    for i in range(num_layers_in_nn):
        layer_set = []
        for j in range(len(nn_layers)):
            layer_set.append(nn_layers[j][i])
        layer_datasets.append(torch.stack(layer_set, 0))
    return layer_datasets


def layer_wise_dataset(nn_dataset: list[list]):
    """
    Returns a tuple of lists containing layerwise weight & bias datasets.
    """
    weight_dataset, bias_dataset = [], []
    for i in range(len(nn_dataset)):
        weights, biases = split_weights_biases(nn_dataset[i])
        weight_dataset.append(weights)
        bias_dataset.append(biases)

    layerwise_weight_datasets = tensor_dataset_layer_wise(len(weights), weight_dataset)
    layerwise_bias_datasets = tensor_dataset_layer_wise(len(weights), bias_dataset)
    return layerwise_weight_datasets, layerwise_bias_datasets

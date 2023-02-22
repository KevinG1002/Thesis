import torch
import torch.nn as nn
import copy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def sample_dict_to_module(base_model: nn.Module, sample_dict: dict):
    base_model = base_model.to(DEVICE)
    new_nn = copy.deepcopy(base_model)
    new_state_dict = dict.fromkeys(new_nn.state_dict())
    weight_keys = [k for k in sample_dict.keys() if k.startswith("w_")]
    bias_keys = [k for k in sample_dict.keys() if k.startswith("b_")]
    b_idx = 0
    w_idx = 0
    for idx, layer in enumerate(base_model.state_dict().keys()):
        if idx % 2 == 0:
            new_state_dict[layer] = sample_dict[weight_keys[w_idx]]
            w_idx += 1
        else:
            new_state_dict[layer] = sample_dict[bias_keys[b_idx]]
            b_idx += 1

    new_nn.load_state_dict(new_state_dict)
    return new_nn

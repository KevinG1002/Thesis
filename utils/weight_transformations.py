import torch
from itertools import combinations
from ast import literal_eval
import copy
import torch.nn as nn
from models.mlp import MLP, SimpleMLP


def nn_to_2d_tensor(nn: nn.Module) -> torch.Tensor:
    concatenated_weights = torch.concat([param.flatten() for param in nn.parameters()])
    n = len(concatenated_weights)
    l, w = width_and_height_algo(n)
    weight_tensor = concatenated_weights.reshape(l, w)
    return weight_tensor


def tensor_to_nn(x: torch.Tensor, base_nn: nn.Module) -> nn.Module:
    """
    Restores a neural network with its hierarchy, layers and bias vectors from a torch tensor.
    This function populates a state_dict with the re-structured weights.
    """
    new_nn = copy.deepcopy(base_nn)
    new_state_dict = dict.fromkeys(new_nn.state_dict())
    assert (
        new_nn is not base_nn
    ), "new_nn points to the same object as base_nn. These are identical"
    x_flattened = x.flatten()
    layer_dims = [param.flatten().size()[0] for param in base_nn.parameters()]
    # start = 0
    curr_idx = 0
    for idx, layer in enumerate(base_nn.state_dict().keys()):
        curr_layer_len = layer_dims[idx]
        structured_layer = torch.gather(
            x_flattened,
            -1,
            torch.arange(curr_idx, curr_idx + curr_layer_len).type(torch.int64),
        ).view(base_nn.state_dict()[layer].size())
        curr_idx += curr_layer_len
        new_state_dict[layer] = structured_layer
    new_nn.load_state_dict(new_state_dict)
    for params_1, params_2 in zip(base_nn.parameters(), new_nn.parameters()):
        # assert (params_1 != params_2).any(), "Parameters haven't changed."
        pass
    return new_nn


def print_weight_dims(base_nn: nn.Module):
    for param in base_nn.parameters():
        print(param.size())


def prime_decomposition(number: int, prime_factors: list = []) -> list[int]:
    """
    number (int): Denotes the length of the flattened tensor containing the weights of the MLP network.
    prime_factors (list): Empty list used to append prime factors found through recursion.
    """
    divider = 2
    # 10
    while divider**2 <= number:
        if number % divider == 0:
            prime_factors.append(divider)
            return prime_decomposition(int(number / divider), prime_factors)
        else:
            divider += 1
    prime_factors.append(number)
    return prime_factors


def bipartition_list(number_list: list):
    """
    number_list (list): list containing prime factors associated with some number

    Returns a list of unique tuples of bipartitions of these prime factors. eg. If number_list is [1,2,3]
    Then it will return: [([1], [2,3]), ([2], [1,3]), ([3], [1,2])]
    """
    n = len(number_list)
    n_partitions = 2 ** (n - 1) - 1
    partition_list = []
    for i in range(1, n_partitions):
        part_1 = []
        part_2 = []
        bin_rep = bin(i)[2:].zfill(n)
        for j in range(n):
            if bin_rep[j] == "0":
                part_1.append(number_list[j])
            else:
                part_2.append(number_list[j])
        partition_list.append((part_1, part_2))
    partitions = {str(partition_list[k]): k for k in range(len(partition_list))}
    partition_list = [literal_eval(k) for k in partitions.keys()]
    return partition_list


def cum_prod(number_list: list):
    """
    Helper function to calculate cumulative product of elements in a list
    """
    prod = 1
    for i in range(len(number_list)):
        prod = prod * number_list[i]

    return prod


def width_and_height_algo(number: int):
    """
    number (int): Denotes the length of the flattened tensor containing the weights of the MLP network.
    Returns the optimal heigh and width used to tranform a flattened weight tensor into a 2d matrix with a
    height to width ratio nearing 1.
    """
    prime_factors = prime_decomposition(number, [])
    prime_factor_bipartitions = bipartition_list(prime_factors)
    len_width_tuples = [
        (cum_prod(x[0]), cum_prod(x[1])) for x in prime_factor_bipartitions
    ]
    optimal_len, optimal_width = [
        len_width_tuples[i]
        for i, elem in enumerate(len_width_tuples)
        if abs(elem[0] - elem[1]) == min([abs(x[0] - x[1]) for x in len_width_tuples])
    ][0]
    return optimal_len, optimal_width

import torch

import copy
import torch.nn as nn


def nn_to_2d_tensor(nn: nn.Module) -> torch.Tensor:
    concatenated_weights = torch.concat([param.flatten() for param in nn.parameters()])
    concatenated_weights = concatenated_weights.view(103 * 3, 59 * 6)
    print(concatenated_weights.size())


def tensor_to_nn(x: torch.Tensor, base_nn: nn.Module) -> nn.Module:
    new_nn = copy.deepcopy(base_nn)
    x_flattened = x.flatten()
    layer_dims = [param.flatten().size()[0] for param in base_nn.parameters()]
    # start = 0
    for idx, layer in enumerate(base_nn.state_dict().keys()):
        curr_layer_len = layer_dims[idx]
        new_nn[layer] = torch.gather(x_flattened, 0, curr_layer_len).view(
            base_nn.state_dict()[layer].size()
        )


def print_weight_dims(base_nn: nn.Module):
    for param in base_nn.parameters():
        print(param.size())

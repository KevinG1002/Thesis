# Script of helper functions that will be used to convert weight tensors to a graph representation for graph networks;
from utils.profile import profile
import torch
import torch.nn as nn
import networkx as nx
import torch_geometric as pyg
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import Data
import itertools
from models.mlp import MLP
import matplotlib.pyplot as plt


def weight_tensor_to_graph(model: torch.nn.Module):
    layer_widths = get_layer_widths(model)
    G = mlp_tensor_to_graph(model, *layer_widths)
    return networkx_to_pygeometric(G)


# @profile
def mlp_tensor_to_graph(model, *subset_sizes):
    # model = MLP(784, 10)
    number_of_weights = 0
    for layer in model.parameters():
        number_of_weights += layer.numel()

    extents = nx.utils.pairwise(
        itertools.accumulate((0,) + subset_sizes)
    )  # creates simple markov chain graph of accumated sum of nodes in graph: (0, 784) -> (784, 784+128=912) -> (912, 912 + 64 = 976) -> (976, 976 + 10=986)
    # print(*itertools.accumulate((0,) + subset_sizes))
    # print(*nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes)))
    layers = [
        range(start, end) for start, end in extents
    ]  # list of layer ranges (range objects)
    layer_params = [
        weight_m for weight_m in model.parameters() if len(weight_m.size()) > 1
    ]
    G = nx.DiGraph()
    for (i, layer) in enumerate(layers):
        G.add_nodes_from(layer, layer=i)
    for idx, (layer1, layer2) in enumerate(nx.utils.pairwise(layers)):

        # print(list((zip(*edges, layer_params[idx]))))
        list_of_edges = [edge_tup for edge_tup in itertools.product(layer1, layer2)]
        weighted_edges = [
            weighted_edge
            for weighted_edge in [
                (
                    (a, b, c)
                    for (a, b), c in zip(
                        list_of_edges, layer_params[idx].mT.flatten().detach()
                    )
                )
            ][0]
        ]
        G.add_weighted_edges_from(weighted_edges)
        # print(list(*weighted_edges[0:256]))
        # exit(0)
    return G


# @profile
def networkx_to_pygeometric(G: nx.Graph) -> Data:
    pyg_graph = from_networkx(G, group_edge_attrs=["weight"])
    pyg_graph.edge_weight = pyg_graph.edge_attr
    return pyg_graph


def get_layer_widths(model: nn.Module):
    layer_widths = []
    input_size = None

    # Iterate over all layers in the model
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            # For Linear layers, append the output size as a layer width
            layer_widths.append(layer.out_features)
            if input_size is None:
                # If this is the first Linear layer encountered, set the input size
                input_size = layer.in_features

    # Append the input size as the first layer width
    layer_widths.insert(0, input_size)

    return layer_widths


def run():
    mlp_layers = [784, 128, 64, 10]
    nn = MLP()
    print(get_layer_widths(nn))
    G = mlp_tensor_to_graph(nn, *mlp_layers)
    print(G.edges[975, 985]["weight"])
    pyg_G = networkx_to_pygeometric(G)
    print(pyg_G["edge_attr"].size())
    return pyg_G


def plot_mlp_graph(G: nx.Graph):
    pos = nx.multipartite_layout(G, subset_key="layer")
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=0.5)
    nx.draw_networkx_edges(G, pos, width=0.001)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()

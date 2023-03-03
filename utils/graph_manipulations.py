# Script of helper functions that will be used to convert weight tensors to a graph representation for graph networks;
from utils.profile import profile
import copy
import torch
import torch.nn as nn
import networkx as nx
import torch_geometric as pyg
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import Data
import itertools
from models.mlp import MLP
import matplotlib.pyplot as plt

torch.manual_seed(0)


def weight_tensor_to_graph(model: torch.nn.Module):
    layer_widths = get_layer_widths(model)
    G = mlp_tensor_to_graph(model, *layer_widths)
    return networkx_to_pygeometric(G)


# @profile
def mlp_tensor_to_graph(model: nn.Module, *subset_sizes):
    # model = MLP(784, 10)
    number_of_weights = 0
    print(subset_sizes)
    for layer in model.parameters():
        number_of_weights += layer.numel()
    extents = nx.utils.pairwise(
        itertools.accumulate((0,) + subset_sizes)
    )  # creates simple markov chain graph of accumated sum of nodes in graph: (2, 786) -> (786, 786+128=914) -> (912, 912 + 64 = 976) -> (976, 976 + 10=986)
    # print(*itertools.accumulate((0,) + subset_sizes))
    # print(*nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes)))
    # print(list(zip(*(zip(*extents)))))
    # layers = [
    #     range(start + 1, end + 2) if start != 0 else range(start, end)
    #     for start, end in extents
    # ]  # list of layer ranges (range objects)
    unpacked_extents = [(start, end) for start, end in extents]
    print(unpacked_extents)
    layers = []
    bias_node_idx = []
    print(len(unpacked_extents))
    prev_end = 0
    delta = 0
    for i in range(len(unpacked_extents)):
        if i == 0:
            cur_start, cur_end = unpacked_extents[i]
            layers.append(range(cur_start, cur_end))
            prev_end = cur_end
        else:
            cur_start, cur_end = unpacked_extents[i]
            delta = cur_end - cur_start
            new_start = prev_end + 1
            new_end = new_start + delta
            layers.append(range(new_start, new_end))
            prev_end = new_end
    print(layers)
    # exit(0)
    layer_params = [
        weight_m for weight_m in model.parameters() if len(weight_m.size()) > 1
    ]
    layer_biases = [param for param in model.parameters() if len(param.size()) < 2]
    print(layer_biases[0])
    G = nx.DiGraph()
    for (i, layer) in enumerate(layers):
        G.add_nodes_from(layer, layer=i)
        if i < len(layers) - 1:
            G.add_node((layer[-1] + 1,), layer=i)
            bias_node_idx.append(layer[-1] + 1)
            # Add MLP Weights
    print(G.number_of_nodes())
    print("Bias node idx:", bias_node_idx)
    for idx, (layer1, layer2) in enumerate(nx.utils.pairwise(layers)):
        list_of_edges = [edge_tup for edge_tup in itertools.product(layer1, layer2)]
        print(len(list_of_edges))
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

    for idx in range(len(bias_node_idx)):
        bias_tup = (bias_node_idx[idx],)
        target_range = layers[idx + 1]
        print(bias_tup)
        print(target_range)
        bias_edges = [
            edge_tup for edge_tup in itertools.product(bias_tup, target_range)
        ]
        weighted_edges = [
            weighted_edge
            for weighted_edge in [
                (
                    (a, b, c)
                    for (a, b), c in zip(
                        bias_edges, layer_biases[idx].flatten().detach()
                    )
                )
            ][0]
        ]
        G.add_weighted_edges_from(weighted_edges)

    print(G.edges(784, "weight"), len(G.edges(784)))
    print(G)

    # exit(0)

    # TODO: Add Bias Edges + Weights
    # exit(0)

    # print(list(*weighted_edges[0:256]))
    # exit(0)
    return G


def torch_geometric_to_networkx(G_Line_Graph: Data):
    """
    Converts torch_geometric data to NetworkX graph
    """
    pass


def networkx_to_torch_nn(G: nx.DiGraph, base_nn: nn.Module):
    """
    Converts NetworkX graph to Pytorch nn.Module for inference on testing set from generated weights.
    """
    pass


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
    mlp_layers = get_layer_widths(nn)
    G = mlp_tensor_to_graph(nn, *mlp_layers)
    # L_G: nx.Graph = nx.line_graph(G, nx.DiGraph())
    # print(L_G.number_of_edges(), L_G.number_of_nodes())

    # for node in L_G.nodes():
    #     orig_u, orig_v = node
    #     weight = G[orig_u][orig_v]["weight"]
    #     L_G.nodes[node]["weight"] = weight.item()
    # pyg_G = networkx_to_pygeometric(G)
    return G


def plot_mlp_graph(G: nx.Graph):
    pos = nx.multipartite_layout(G, subset_key="layer")
    # pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=0.5)
    nx.draw_networkx_edges(G, pos, width=0.001)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    g = run()
    # plot_mlp_graph(g)

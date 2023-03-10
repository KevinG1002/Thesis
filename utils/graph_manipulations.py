# Script of helper functions that will be used to convert weight tensors to a graph representation for graph networks;
from utils.profile import profile
import copy
import torch
import torch.nn as nn
import networkx as nx
import torch_geometric as pyg
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
import itertools
from models.mlp import MLP
import matplotlib.pyplot as plt

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def weight_tensor_to_graph(model: torch.nn.Module):
    """
    Takes nn.Module weights & biases and returns PyGeometric Graph representation of these weights.
    """
    G = mlp_tensor_to_line_graph(model)
    return networkx_to_pygeometric(G)


def convert_PyG_LineGraph_to_Nx_OG_Graph(Line_G: Data, Ref_Line_G: nx.DiGraph):
    """
    Converts Data Line Graph object to a Nx.DiGraph(), then creates a new Nx.LineGraph with the weights from Data.
    """

    for node, attrs in Ref_Line_G.nodes.data():
        attrs.clear()

    # print(Ref_Line_G.nodes(data=True))  # output: [(1, {}), (2, {})]
    # print(Ref_Line_G.nodes(0))
    # print(Line_G.node_stores)
    restored_graph = nx.DiGraph()
    for idx, node_attrs in enumerate(Ref_Line_G.nodes.data()):
        node, _ = node_attrs
        u, v = node
        # print(node)
        # print(u, v)
        # exit(0)
        restored_graph.add_edge(u, v, weight=Line_G.x.squeeze()[idx])
    H = nx.DiGraph()
    H.add_nodes_from(sorted(restored_graph.nodes(data=True)))
    H.add_weighted_edges_from(
        sorted(restored_graph.edges.data("weight"), key=lambda tup: tup[0])
    )
    # return restored_graph
    return H


def get_basis_graph(model: torch.nn.Module, layer_widths):
    """
    Find a way to construct graph without weighted edges, just edges.
    """
    pass


def restore_graph_from_nx_line_graph(
    base_graph: nx.DiGraph, line_graph: nx.DiGraph, node_features: torch.Tensor
):
    """
    Restores original NX DiGraph from an NX directed Line graph where the nodes are *tuples of nodes* describing the edge it is representing.
    """
    print(line_graph.nodes.data("weight"))
    restored_graph = nx.DiGraph()
    # print("Number of nodes", len(restored_graph.nodes()))
    # print("Number of edges", len(restored_graph.edges()))
    for (u, v), weight in line_graph.nodes("weight"):
        # print("Source Node", u)
        # print("Destination Node", v)
        # print("Weight", weight)
        restored_graph.add_edge(u, v, weight=weight)

    # for weight_a in restored_graph.edges.data("weight"):
    #     if not (weight_a):
    #         print(weight_a)
    # print(restored_graph.edges.data("weight"))
    restored_sorted_by_first = sorted(
        restored_graph.edges.data("weight"), key=lambda tup: tup[0]
    )
    original_sorted_by_first = sorted(
        base_graph.edges.data("weight"), key=lambda tup: tup[0]
    )
    for weight_a, weight_b in zip(restored_sorted_by_first, original_sorted_by_first):
        if not (weight_a == weight_b):
            print(weight_a, weight_b)
        else:
            print(weight_a, weight_b)

    return restored_graph


# @profile
def mlp_tensor_to_graph(model: nn.Module, *subset_sizes):
    """
    From a nn.Module takes the weights & biases and returns a weighted NetworkX graph.
    """
    # model = MLP(784, 10)
    number_of_params = 0
    for layer in model.parameters():
        number_of_params += layer.numel()
    # print(number_of_params)
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
    # print(unpacked_extents)
    layers = []
    bias_node_idx = []
    # print(len(unpacked_extents))
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
    # print(layers)
    # exit(0)
    layer_params = [
        weight_m for weight_m in model.parameters() if len(weight_m.size()) > 1
    ]
    layer_biases = [param for param in model.parameters() if len(param.size()) < 2]
    # print(layer_biases[0])
    G = nx.DiGraph()
    for (i, layer) in enumerate(layers):
        G.add_nodes_from(layer)
        if i < len(layers) - 1:
            G.add_node(
                layer[-1] + 1,
            )  # Temporarily removed "layer" attribute when adding node.
            bias_node_idx.append(layer[-1] + 1)
            # Add MLP Weights
    # print(G.number_of_nodes())
    # print("Bias node idx:", bias_node_idx)
    for idx, (layer1, layer2) in enumerate(nx.utils.pairwise(layers)):
        list_of_edges = [edge_tup for edge_tup in itertools.product(layer1, layer2)]
        weighted_edges = [
            weighted_edge
            for weighted_edge in [
                (
                    (a, b, c.item())
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
        # print(bias_tup)
        # print(target_range)
        bias_edges = [
            edge_tup for edge_tup in itertools.product(bias_tup, target_range)
        ]
        weighted_edges = [
            weighted_edge
            for weighted_edge in [
                (
                    (a, b, c.item())
                    for (a, b), c in zip(
                        bias_edges, layer_biases[idx].flatten().detach()
                    )
                )
            ][0]
        ]
        G.add_weighted_edges_from(weighted_edges)

    # print(G.edges(784, "weight"), len(G.edges(784)))
    # print(G)

    # exit(0)

    # TODO: Add Bias Edges + Weights
    # exit(0)

    # print(list(*weighted_edges[0:256]))
    # exit(0)
    # Line_G: nx.DiGraph = nx.line_graph(G, nx.DiGraph())
    # for node in Line_G.nodes():
    #     orig_u, orig_v = node
    #     weight = G[orig_u][orig_v]["weight"]
    #     Line_G.nodes[node]["weight"] = weight

    ## Sort node order and edge connection order.
    H = nx.DiGraph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_weighted_edges_from(sorted(G.edges.data("weight"), key=lambda tup: tup[0]))
    # return G
    return H
    # return Line_G


def mlp_tensor_to_line_graph(model: nn.Module):
    """
    Converts MLP nn.Module to Line Graph. Purpose of this function is to apply this transformation to graphs before converting
    them to PyGeometric objects. We do this instead of using the LineGraph transformation proposed by PyG because we don't
    understand how PyG maps a weighted edge graph to a line graph.
    """
    layer_widths = get_layer_widths(model)
    G = mlp_tensor_to_graph(model, *layer_widths)

    Line_G: nx.DiGraph = nx.line_graph(G, nx.DiGraph())
    for node in Line_G.nodes():
        orig_u, orig_v = node
        weight = G[orig_u][orig_v]["weight"]
        Line_G.nodes[node]["weight"] = weight
    return Line_G


def torch_geometric_to_networkx(G_Line_Graph: Data):
    """
    Converts torch_geometric data object to NetworkX graph objext
    """
    g = to_networkx(G_Line_Graph, to_undirected=False, node_attrs=["weight"])
    print(g)
    return g


def networkx_to_torch_nn(G: nx.DiGraph, base_nn: nn.Module):
    """
    Converts NetworkX graph to Pytorch nn.Module for inference on testing set from generated weights.
    """
    # Check if line graph
    # layer_widths = get_layer_widths(base_nn)
    # ref_G = mlp_tensor_to_graph(base_nn, *layer_widths)
    # print(G.number_of_nodes(), ref_G.number_of_nodes())
    # if ref_G.number_of_nodes() != G.number_of_nodes():
    #     G = nx.inverse_line_graph(G)
    number_of_params = 0
    for layer in base_nn.parameters():
        number_of_params += layer.numel()
    # x = x.to(DEVICE)
    base_nn = base_nn.to(DEVICE)
    new_nn = copy.deepcopy(base_nn)
    new_state_dict = dict.fromkeys(new_nn.state_dict())
    # num_nodes = G.number_of_nodes()
    # weights = torch.tensor([d["weight"] for u, v, d in sorted_edges])
    weights = torch.tensor([G[u][v]["weight"] for u, v in G.edges()])
    # print("Bias in Graph", G.edges(784, "weight"))
    # print("Bias in State Dict", new_nn.state_dict()["fc_1.bias"])
    # print("Next Layer Weights in State Dict", new_nn.state_dict()["fc_2.weight"])
    # print("Assumed Bias in weights", weights[100352:100500])
    # print((weights[0:100352] == new_nn.state_dict()["fc_1.weight"].mT.flatten()).all())
    # print(
    #     (
    #         weights[0:100352].view(new_nn.state_dict()["fc_1.weight"].mT.size()).mT
    #         == new_nn.state_dict()["fc_1.weight"]
    #     ).all()
    # )
    assert (
        len(weights) == number_of_params
    ), "Missmatch between number of params taken from graph and number of params in nn.Module."
    weight_dims = [
        param.flatten().size()[0]
        for param in base_nn.parameters()
        if len(param.size()) > 1
    ]
    bias_dims = [
        param.flatten().size()[0]
        for param in base_nn.parameters()
        if len(param.size()) < 2
    ]
    # print("Weight dims", weight_dims)
    # print("Bias dims", bias_dims)
    layer_dims = (
        weight_dims + bias_dims
    )  # Concatenate list of bias dims to end of list of weight dims and iterate over indices because of how weighted edges were added to graph. Can probably clean this.
    # print("Layer Dims", layer_dims)
    curr_idx = 0
    bias_start_idx = sum(weight_dims)

    ### PREVIOUS APPROACH ###
    # print(bias_start_idx)
    # for idx, layer in enumerate(base_nn.state_dict().keys()):
    #     if "weight" in layer:
    #         curr_layer_len = layer_dims[idx // 2]
    #         # print("Curr Layer", layer)
    #         # print("Param Range", curr_idx, curr_idx + curr_layer_len)
    #         # print(curr_layer_len)
    #         structured_layer = torch.gather(
    #             weights,
    #             -1,
    #             torch.arange(curr_idx, curr_idx + curr_layer_len)
    #             .type(torch.int64)
    #             .to(DEVICE),
    #         ).view(base_nn.state_dict()[layer].mT.size())
    #         new_state_dict[layer] = structured_layer.mT
    #         curr_idx += curr_layer_len
    #     else:
    #         cur_bias_len = bias_dims[(idx - 1) // 2]

    #         # print("Curr Bias Layer", layer)
    #         # print("Param Range", bias_start_idx, bias_start_idx + cur_bias_len)
    #         structured_layer = torch.gather(
    #             weights,
    #             -1,
    #             torch.arange(bias_start_idx, bias_start_idx + cur_bias_len)
    #             .type(torch.int64)
    #             .to(DEVICE),
    #         ).view(base_nn.state_dict()[layer].size())
    #         bias_start_idx += cur_bias_len
    #         new_state_dict[layer] = structured_layer

    layer_dims = [param.flatten().size()[0] for param in base_nn.parameters()]
    # start = 0
    curr_idx = 0
    for idx, layer in enumerate(base_nn.state_dict().keys()):
        curr_layer_len = layer_dims[idx]
        if idx % 2 != 0:
            structured_layer = torch.gather(
                weights,
                -1,
                torch.arange(curr_idx, curr_idx + curr_layer_len)
                .type(torch.int64)
                .to(DEVICE),
            ).view(base_nn.state_dict()[layer].size())
            curr_idx += curr_layer_len
            new_state_dict[layer] = structured_layer
        else:
            structured_layer = torch.gather(
                weights,
                -1,
                torch.arange(curr_idx, curr_idx + curr_layer_len)
                .type(torch.int64)
                .to(DEVICE),
            ).view(base_nn.state_dict()[layer].mT.size())
            curr_idx += curr_layer_len
            new_state_dict[layer] = structured_layer.mT
    new_nn.load_state_dict(new_state_dict)

    # # print(structured_layer)
    # # print(base_nn.state_dict()[layer])
    # # curr_idx += curr_layer_len
    # # print(layer)
    # if "weight" in layer:
    #     print("Hello")

    # else:
    #     print("Goodbye")
    #     new_state_dict[layer] = structured_layer
    # new_nn.load_state_dict(new_state_dict)
    return new_nn


# @profile
def networkx_to_pygeometric(G: nx.Graph) -> Data:
    """
    Converts NetworkX graph to a PyGeometric graph (Data type).
    """
    # pyg_graph = from_networkx(G, group_edge_attrs=["weight"])
    # pyg_graph.edge_weight = pyg_graph.edge_attr
    pyg_graph = from_networkx(G, group_node_attrs=["weight"])
    return pyg_graph


def get_layer_widths(model: nn.Module) -> list:
    """
    Given a nn.Module, returns a list of its layer widths.
    """
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


def pygeometric_to_nn(G: Data, base_model: nn.Module) -> nn.Module:
    """
    Principal function used after sampling from GVAE to convert PyG graph to nn.Module.
    """
    # layer_widths = get_layer_widths(
    #     base_model
    # )  # get layer widths of our base model for reconstruction of our NN Module.
    # ref_G = mlp_tensor_to_graph(base_model, *layer_widths)  # get reference nX graph.
    # ref_Line_G = nx.line_graph(ref_G, nx.DiGraph)
    # # nx_Line_G = torch_geometric_to_networkx(G)  # convert PyG LineGraph to nX graph.
    # nx_G = restore_graph_from_nx_line_graph(
    #     ref_G, ref_Line_G, G.x.squeeze()
    # )  # restores original weighted graph from line graph.
    # return networkx_to_torch_nn(nx_G, base_model)
    Ref_Line_G = mlp_tensor_to_line_graph(base_model)
    restored_G = convert_PyG_LineGraph_to_Nx_OG_Graph(G, Ref_Line_G)
    nn = networkx_to_torch_nn(restored_G, base_model)
    return nn


def run():
    mlp_layers = [784, 128, 64, 10]
    nn = MLP()
    nn.load_state_dict(
        torch.load(
            "/Users/kevingolan/Documents/Coding_Assignments/Thesis/datasets/model_dataset_MNIST/models/mlp_mnist_model_1.pth",
            map_location=DEVICE,
        )
    )
    PyG_G = weight_tensor_to_graph(nn)
    restored_nn = pygeometric_to_nn(PyG_G, MLP())

    # print(get_layer_widths(nn))
    # G: nx.DiGraph = mlp_tensor_to_graph(nn, *mlp_layers)
    # print(G.edges.data("weight"))
    # sorted_nx_G_weights = torch.tensor(
    #     [d for _, _, d in sorted(G.edges.data("weight"), key=lambda x: x[0])]
    # )
    # actual_nx_G_weights = torch.tensor([d for _, _, d in G.edges.data("weight")])
    # assert (
    #     sorted_nx_G_weights == actual_nx_G_weights
    # ).all(), "Graph weights are not sorted according to ascending order of source nodes in DiGraph."
    # # H = nx.DiGraph()
    # # H.add_nodes_from(sorted(G.nodes(data=True)))
    # # H.add_edges_from(G.edges(data=True))
    # Line_G = mlp_tensor_to_line_graph(nn)

    # # Line_G = nx.line_graph(G, nx.DiGraph())
    # # print(Line_G.nodes(data=False))
    # # exit()
    # PyG_Line_G: Data = from_networkx(Line_G, group_node_attrs=["weight"])

    # assert (
    #     PyG_Line_G.x.squeeze()
    #     == torch.tensor([d for _, d in Line_G.nodes.data("weight")])
    # ).all(), (
    #     "Line Graph from PyG has weights not aligned with NetworkX Line Graph weights."
    # )
    # restored_G = convert_PyG_LineGraph_to_Nx_OG_Graph(PyG_Line_G, Line_G)
    # print(
    #     all(
    #         [
    #             x == y
    #             for x, y in zip(
    #                 list(G.edges.data("weight")), list(restored_G.edges.data("weight"))
    #             )
    #         ]
    #     )
    # )
    # restored_nn = networkx_to_torch_nn(restored_G, copy.deepcopy(nn))
    assert all(
        [
            (param1 == param2).all()
            for param1, param2 in zip(nn.parameters(), restored_nn.parameters())
        ]
    ), "Neural Network Params do not match."
    # Line_G: nx.DiGraph = nx.line_graph(G, nx.DiGraph)
    # for node in Line_G.nodes():
    #     orig_u, orig_v = node
    #     weight = G[orig_u][orig_v]["weight"]
    #     Line_G.nodes[node]["weight"] = weight  # .item()
    # # recovered_G = restore_graph_from_line_graph(G, Line_G)
    # # nx_line_G_weights = torch.tensor(
    # #     [d for _, d in sorted(Line_G.nodes.data("weight"), key=lambda x: x[0])]
    # )
    # nx_line_G_weights = torch.tensor([d for _, d in Line_G.nodes.data("weight")])
    # PyG_transform = LineGraph(False)
    # PyG_LineG = from_networkx(Line_G, group_node_attrs=["weight"])
    # print((PyG_LineG.x.squeeze() == nx_line_G_weights).all())
    # missmatches = 0
    # for k in range(len(nx_line_G_weights)):
    #     if PyG_LineG.x.squeeze()[k] != nx_line_G_weights[k]:
    #         missmatches += 1
    #         # print(k, PyG_LineG.x.squeeze()[k], nx_line_G_weights[k])
    #     else:
    #         print(k, PyG_LineG.x.squeeze()[k], nx_line_G_weights[k])
    # print("Number of mismatches", missmatches)
    # print(list(G.edges.data("weight"))[0:50])
    # a = list(G.edges.data("weight"))
    # b = list(recovered_G.edges.data("weight"))

    # print()
    # print(list(recovered_G.edges.data("weight"))[0:50])

    # print(
    #     [
    #         x == y
    #         for x, y in zip(
    #             list(G.edges.data("weight")), list(recovered_G.edges.data("weight"))
    #         )
    #     ]
    # )

    # print(Line_G.number_of_nodes())
    # print(list(Line_G.nodes)[-100:None:None])
    # restored_nn = networkx_to_torch_nn(G, copy.deepcopy(nn))
    # print(
    #     [
    #         (param1 == param2).all()
    #         for param1, param2 in zip(nn.parameters(), restored_nn.parameters())
    #     ]
    # )
    # L_G: nx.Graph = nx.line_graph(G, nx.DiGraph())
    # print(L_G.number_of_edges(), L_G.number_of_nodes())

    # for node in L_G.nodes():
    #     orig_u, orig_v = node
    #     weight = G[orig_u][orig_v]["weight"]
    #     L_G.nodes[node]["weight"] = weight.item()
    # pyg_G = networkx_to_pygeometric(G)
    return


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

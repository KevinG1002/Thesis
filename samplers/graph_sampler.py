import torch_geometric as PyG
import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.utils import remove_self_loops, to_dense_adj, to_networkx, subgraph
from torch_geometric.utils.sparse import to_edge_index
from torch_geometric.data import Data
from torch_geometric.sampler import BaseSampler
from torch_geometric.sampler.base import (
    NodeSamplerInput,
    HeteroSamplerOutput,
    SamplerOutput,
    EdgeSamplerInput,
)
import numpy as np
from torch_geometric.loader import NeighborLoader
from typing import Any, Dict, List, Optional, Union


class GVAE_Sampler(nn.Module):
    def __init__(
        self,
        measure: str,
        alpha: Union[float, int],
        edge_index: torch.Tensor,
        subgraph_num_nodes: int,
    ):
        super(GVAE_Sampler, self).__init__()
        self.subgraph_num_nodes = subgraph_num_nodes
        self.proba = self.node_sampling_distribution(measure, alpha, edge_index)

    def forward(self, x, edge_index):
        adj = to_dense_adj(edge_index)
        subgraph_nodes, subgraph_edge_index, subgraph_adj = self.node_sampling(
            adj, self.proba, self.subgraph_num_nodes
        )
        sampled_subgraph_edge_index, _ = subgraph(subgraph_nodes, edge_index)
        sampled_subgraph_x = x[subgraph_nodes]
        return sampled_subgraph_x, sampled_subgraph_edge_index

    def node_sampling(self, adj, distribution, n_node_samples, replace=False):
        """
        Sample a subgraph from a given node-level distribution
            :param adj: sparse adjacency matrix of the graph
            :param distribution: p_i distribution, from get_distribution()
            :param nb_node_samples: size (nb of nodes) of the sampled subgraph
            :param replace: whether to sample nodes with replacement
            :return: nodes from the sampled subgraph, and subgraph adjacency matrix
        """

        # Sample nb_node_samples nodes, from the pre-computed distribution
        sampled_nodes = np.random.choice(
            adj.shape[0], size=n_node_samples, replace=replace, p=distribution
        )
        # Sparse adjacency matrix of sampled subgraph
        sampled_adj = adj[sampled_nodes, :][:, sampled_nodes]
        sampled_adj += np.eye(sampled_adj.shape[0])
        sampled_adj = torch.tensor(sampled_adj)
        # In tuple format (useful for optimizers)
        sampled_edge_index = to_edge_index(sampled_adj)
        return sampled_nodes, sampled_edge_index, sampled_adj

    def node_sampling_distribution(
        self, measure: str, alpha: float, edge_index: torch.Tensor
    ):
        """
        Dictionary of methods from which we can assign probabilities to nodes in our graph getting picked (4 measures in total).
        Function taken and extended from: https://github.com/deezer/fastgae/blob/master/fastgae/sampling.py
        - SignalStrength: Sample nodes based on signal strength exhibited node weights + neighbors. Basically, sample nodes
        that influence output the most (our own).
        - Core: Core-base node sampling taken from FastGAE (https://arxiv.org/pdf/2002.01910.pdf)
        - Degree: Degree-based node sampling taken from FastGAE (https://arxiv.org/pdf/2002.01910.pdf)
        - Uniform: Every node can be chosen uniformy at random.

        :param measure: node importance measure, among 'degree', 'core', 'uniform'
        :param alpha: alpha scalar hyperparameter for degree and core sampling (most be positive and real)
        :param G_adj: sparse adjacency matrix of the graph
        :return: list of p_i probabilities of all nodes
        """
        if measure == "degree":
            # Degree-based distribution (node degree = sum along 1 axis of adjacency matrix)
            G_adj = to_dense_adj(edge_index)
            proba = torch.pow(G_adj.sum(dim=0), torch.tensor(alpha)).tolist()[0]
        elif measure == "core":
            # Core-based distribution
            # G = remove_self_loops(edge_index)
            G_no_loops = remove_self_loops(edge_index)
            G_adj = to_dense_adj(G_no_loops)
            G_adj_nx = to_networkx(G_adj)
            proba = np.power(list(nx.core_number(G_adj_nx).values()), alpha)
        elif measure == "uniform":
            # Uniform distribution
            G_adj = to_dense_adj(edge_index)
            proba = np.ones(G_adj.shape[0])
        else:
            raise ValueError("Undefined sampling method!")
        # Normalization
        proba = proba / np.sum(proba)
        return proba

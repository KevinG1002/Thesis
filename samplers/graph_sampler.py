import torch
import torch.nn as nn
import tqdm
import networkx as nx
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import remove_self_loops, to_dense_adj, to_networkx, subgraph
from torch_geometric.utils.sparse import to_edge_index
from torch_geometric.data import Data, HeteroData, Batch
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.graph_store import EdgeLayout, GraphStore
from torch_geometric.sampler import BaseSampler
from torch_geometric.loader import NodeLoader, DataLoader
from torch_geometric.sampler.base import (
    NodeSamplerInput,
    HeteroSamplerOutput,
    SamplerOutput,
    EdgeSamplerInput,
)
import numpy as np
from datasets.graph_dataset import GraphDataset
from models.mlp import MLP
import os
from typing import Any, Dict, List, Optional, Union, Tuple


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

    def node_sampling(self, adj, distribution, n_node_samples, replace=True):
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


class SubgraphSampler(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: GraphDataset,
        batch_size: int,
        num_nodes: int,
        measure: str,
        alpha: Union[float, int],
    ):
        pass


class GraphSampler(BaseSampler):
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        measure: str,
        alpha: Union[float, int],
        subgraph_num_nodes: int,
        replace: bool,
        time_attr: Optional[str] = None,
    ):
        self.data_cls = (
            data.__class__ if isinstance(data, (Data, HeteroData)) else "custom"
        )
        self.measure = measure
        self.alpha = alpha
        self.subgraph_num_nodes = subgraph_num_nodes
        self.replace = replace
        self.edge_index = data.edge_index
        self.x = data.x
        if isinstance(data, Data):
            if time_attr is not None:
                self.node_time = data[time_attr]

            # Convert the graph data into a suitable format for sampling.
            self.adj = to_dense_adj(
                self.edge_index,
            )
            # assert isinstance(subgraph_num_nodes, (list, tuple))
        else:
            raise TypeError(
                "Wrong data type for sampling. Make sure you pass in a graph in Data format"
            )
        self.probabilities = self.node_sampling_distribution(
            self.measure, self.alpha, self.edge_index
        )
        self.node_index = torch.arange(self.adj.size()[0])

    def sample_from_nodes(
        self, index: NodeSamplerInput, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.node_sampling(
            index, self.probabilities, self.subgraph_num_nodes, self.replace
        )

    def sample_from_edges(
        self, index: EdgeSamplerInput, **kwargs
    ) -> Union[HeteroSamplerOutput, SamplerOutput]:
        return super().sample_from_edges(index, **kwargs)

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
        return torch.tensor(proba)

    def node_sampling(
        self, node_index: torch.Tensor, distribution, n_node_samples, replace=False
    ):
        """
        Sample a subgraph from a given node-level distribution
            :param adj: sparse adjacency matrix of the graph
            :param distribution: p_i distribution, from get_distribution()
            :param nb_node_samples: size (nb of nodes) of the sampled subgraph
            :param replace: whether to sample nodes with replacement
            :return: nodes from the sampled subgraph, and subgraph adjacency matrix
        """
        print(node_index)
        # Sample nb_node_samples nodes, from the pre-computed distribution
        sampled_nodes = np.random.choice(
            node_index.size()[0], size=n_node_samples, replace=replace, p=distribution
        )
        # Sparse adjacency matrix of sampled subgraph
        sampled_adj = self.adj[sampled_nodes, :][:, sampled_nodes]
        sampled_adj += torch.eye(sampled_adj.shape[0])
        # In tuple format (useful for optimizers)
        sampled_edge_index = to_edge_index(sampled_adj)
        return sampled_nodes, sampled_edge_index, sampled_adj


class GraphLoader(DataLoader):
    def __init__(
        self,
        data: Data,
        batch_size: int,
        num_steps: int = 1,
        sample_coverage: int = 0,
        save_dir: Optional[str] = None,
        log: bool = True,
        **kwargs,
    ):

        # Remove for PyTorch Lightning:
        kwargs.pop("dataset", None)
        kwargs.pop("collate_fn", None)

        assert data.edge_index is not None
        assert "node_norm" not in data
        assert "edge_norm" not in data
        assert not data.edge_index.is_cuda

        self.num_steps = num_steps
        self.__batch_size__ = batch_size
        self.sample_coverage = sample_coverage
        self.save_dir = save_dir
        self.log = log

        self.N = data.num_nodes
        self.E = data.num_edges

        self.adj = SparseTensor(
            row=data.edge_index[0],
            col=data.edge_index[1],
            value=torch.arange(self.E, device=data.edge_index.device),
            sparse_sizes=(self.N, self.N),
        )

        self.data = data

        super().__init__(self, batch_size=1, collate_fn=self.__collate__, **kwargs)

        if self.sample_coverage > 0:
            path = os.path.join(save_dir or "", self.__filename__)
            if save_dir is not None and os.path.exists(path):  # pragma: no cover
                self.node_norm, self.edge_norm = torch.load(path)
            else:
                self.node_norm, self.edge_norm = self.__compute_norm__()
                if save_dir is not None:  # pragma: no cover
                    torch.save((self.node_norm, self.edge_norm), path)

    @property
    def __filename__(self):
        return f"{self.__class__.__name__.lower()}_{self.sample_coverage}.pt"

    def __len__(self):
        return self.num_steps

    def __sample_nodes__(self, batch_size):
        raise NotImplementedError

    def __getitem__(self, idx):
        node_idx = self.__sample_nodes__(self.__batch_size__).unique()
        adj, _ = self.adj.saint_subgraph(node_idx)
        return node_idx, adj

    def __collate__(self, data_list):
        assert len(data_list) == 1
        node_idx, adj = data_list[0]

        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        for key, item in self.data:
            if key in ["edge_index", "num_nodes"]:
                continue
            if isinstance(item, torch.Tensor) and item.size(0) == self.N:
                data[key] = item[node_idx]
            elif isinstance(item, torch.Tensor) and item.size(0) == self.E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        if self.sample_coverage > 0:
            data.node_norm = self.node_norm[node_idx]
            data.edge_norm = self.edge_norm[edge_idx]

        return data

    def __compute_norm__(self):
        node_count = torch.zeros(self.N, dtype=torch.float)
        edge_count = torch.zeros(self.E, dtype=torch.float)

        loader = torch.utils.data.DataLoader(
            self, batch_size=200, collate_fn=lambda x: x, num_workers=self.num_workers
        )

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.N * self.sample_coverage)
            pbar.set_description("Compute GraphSAINT normalization")

        num_samples = total_sampled_nodes = 0
        while total_sampled_nodes < self.N * self.sample_coverage:
            for data in loader:
                for node_idx, adj in data:
                    edge_idx = adj.storage.value()
                    node_count[node_idx] += 1
                    edge_count[edge_idx] += 1
                    total_sampled_nodes += node_idx.size(0)

                    if self.log:  # pragma: no cover
                        pbar.update(node_idx.size(0))
            num_samples += self.num_steps

        if self.log:  # pragma: no cover
            pbar.close()

        row, _, edge_idx = self.adj.coo()
        t = torch.empty_like(edge_count).scatter_(0, edge_idx, node_count[row])
        edge_norm = (t / edge_count).clamp_(0, 1e4)
        edge_norm[torch.isnan(edge_norm)] = 0.1

        node_count[node_count == 0] = 0.1
        node_norm = num_samples / node_count / self.N

        return node_norm, edge_norm


def test():
    dataset = GraphDataset(
        base_model=MLP(),
        root="../datasets/model_dataset_MNIST",
    )
    data = Batch.from_data_list(dataset)
    sampler = GraphSampler(dataset[0], "uniform", 2.5, 2000, False)
    loader = NodeLoader(data, sampler)
    for b in loader:
        print(b)


if __name__ == "__main__":
    test()

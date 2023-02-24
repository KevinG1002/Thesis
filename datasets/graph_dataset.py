import torch
import json
import os
from typing import Callable
import torch_geometric as PyG
from torch_geometric.data import Dataset
from torch_geometric.transforms import LineGraph
import torch_geometric.transforms as T
from models.mlp import MLP
from utils.graph_manipulations import weight_tensor_to_graph
import torch.nn as nn
import copy


class GraphDataset(Dataset):
    def __init__(
        self,
        root: str,
        base_model: nn.Module,
        pre_transform: Callable = None,
        transform: Callable = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = base_model
        self.labels_path = os.path.join(
            root, [file for file in os.listdir(root) if file.endswith(".json")][0]
        )
        with open(self.labels_path, "r") as labels:
            self._model_paths, self._model_labels = map(
                list, zip(*list(json.load(labels).items()))
            )
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        # self.labels = os.path.join([os.listdir(root)])

        # print(self._model_paths, self._model_labels)
        # self.raw_file_names(model_paths)
        # self.labels(model_labels)
        # Layer widths of base model including input and output layer

    def process(self):
        """
        Here you apply your pre-transform function to your dataset samples to transform them into a graph.
        """
        idx = 0
        for raw_path in self.raw_paths:
            data = self.load_model(raw_path)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            os.path.join(self.processed_dir, f"data_{idx}.pt"), map_location=self.device
        )
        return data

    def load_model(self, model_path):
        model = copy.deepcopy(self.base_model)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    @property
    def raw_file_names(self):
        return self._model_paths

    @property
    def processed_file_names(self):
        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
        return [file for file in os.listdir(self.processed_dir) if file.endswith(".pt")]

    @property
    def labels(self):
        return self._model_labels


def run():
    transforms = T.Compose([weight_tensor_to_graph, LineGraph(True)])
    gd = GraphDataset(
        base_model=MLP(),
        root="../datasets/model_dataset_MNIST",
        pre_transform=transforms,
    )
    graph = gd[0]
    print(graph.x.size())
    print(graph.num_edges)
    print(graph.num_nodes)
    print(graph.edge_index.shape)


if __name__ == "__main__":
    run()

import torch
import json
import os
from typing import Callable
import torch_geometric as PyG
from torch_geometric.data import Dataset
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
        print(os.listdir(root))
        self.base_model = base_model
        self.labels_path = os.path.join(
            root, [file for file in os.listdir(root) if file.endswith(".json")][0]
        )
        super(GraphDataset, self).__init__(root, pre_transform, transform)
        # self.labels = os.path.join([os.listdir(root)])
        with open(self.labels_path, "r") as labels:
            self.model_paths, self.model_labels = map(
                list, zip(*list(json.load(labels).items()))
            )

    def process(self):
        """
        Here you apply your pre-transform function to your dataset samples to transform them into a graph.
        """
        idx = 0
        for raw_path in self.raw_paths:
            data = torch.load(raw_path)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))
        return data

    def load_model(self, model_path):
        model = copy.deepcopy(self.base_model)
        model.load_state_dict(torch.load(model_path))
        return model

    @property
    def raw_file_names(self):
        return self.model_paths


def run():
    gd = GraphDataset(root="../datasets/model_dataset_MNIST")


if __name__ == "__main__":
    run()

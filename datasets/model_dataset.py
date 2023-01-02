import torch
import json
import os
import torch.nn as nn
from typing import Callable
from torch.utils.data import Dataset
from utils.weight_transformations import print_weight_dims, nn_to_2d_tensor
from models.mlp import MLP, SimpleMLP


class ModelsDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        model_labels_path: str,
        base_model: nn.Module,
        manipulations: Callable = None,
    ):
        self.root_dir = root_dir
        self.model_labels_path = model_labels_path
        with open(self.model_labels_path, "r") as labels:
            self.model_paths, self.model_labels = map(
                list, zip(*list(json.load(labels).items()))
            )
        self.manipulations = manipulations
        self.base_model = base_model

    def __getitem__(self, index):
        model_path = os.path.join(self.root_dir, self.model_paths[index])
        model = self.load_model(model_path)
        label = self.model_labels[index]
        if self.manipulations:
            model = self.manipulations(model)
        return model, label

    def load_model(self, model_path):
        model = self.base_model
        model.load_state_dict(torch.load(model_path))
        return model

    @property
    def num_samples(self):
        return len(self.model_labels)

    def __len__(self):
        return self.num_samples

    def print_architecture(self):
        print(self.base_model)

    @property
    def flattened_sample_dim(self):
        return sum(
            [
                param.numel()
                for param in self.base_model.parameters()
                if param.requires_grad
            ]
        )


def main():
    dataset = ModelsDataset(
        root_dir="/Users/kevingolan/Documents/Coding_Assignments/Thesis/datasets/model_dataset_MNIST/model_dataset.json",
        model_labels_path="/Users/kevingolan/Documents/Coding_Assignments/Thesis/datasets/model_dataset_MNIST/model_dataset.json",
        base_model=MLP(784, 10),
    )
    dataset.print_architecture()
    print(len(dataset))
    print(dataset.flattened_sample_dim)
    print_weight_dims(dataset.base_model)
    nn_to_2d_tensor(dataset.base_model)


if __name__ == "__main__":
    main()

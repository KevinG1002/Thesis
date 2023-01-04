import torch
import random
import json
import os
import copy
import torch.nn as nn
from typing import Callable
from torch.utils.data import Dataset, DataLoader
from utils.weight_transformations import pad_to, unpad, nn_to_2d_tensor
from models.mlp import MLP, SimpleMLP


class ModelsDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        model_labels_path: str,
        base_model: nn.Module,
        manipulations: Callable = None,
        padding: bool = True,
        original_dataset: str = "MNIST",
    ):
        self.root_dir = root_dir
        self.model_labels_path = model_labels_path
        with open(self.model_labels_path, "r") as labels:
            self.model_paths, self.model_labels = map(
                list, zip(*list(json.load(labels).items()))
            )
        self.manipulations = manipulations
        self.base_model = base_model
        self.padding = padding
        self.original_dataset = original_dataset

    def __getitem__(self, index):
        # model_path = os.path.join(self.model_paths[index])
        loaded_model = self.load_model(self.model_paths[index])
        label = self.model_labels[index]
        if self.manipulations:
            manipulated_model = self.manipulations(loaded_model)
            if self.padding:
                padded_model, self.pads = pad_to(
                    manipulated_model, 16
                )  # pad model representation so that it can be passed through UNet predicting noise.
                return padded_model, label
            else:
                return manipulated_model, label
        return loaded_model, label

    def load_model(self, model_path):
        model = copy.deepcopy(MLP())
        model.load_state_dict(torch.load(model_path))
        return model

    @property
    def num_samples(self):
        return len(self.model_labels)

    def __len__(self):
        return self.num_samples

    def print_architecture(self):
        print(self.base_model)

    def restore_original_tensor(self, tensor: torch.Tensor):
        return unpad(tensor, self.padding_used)

    @property
    def flattened_sample_dim(self):
        return sum(
            [
                param.numel()
                for param in self.base_model.parameters()
                if param.requires_grad
            ]
        )

    @property
    def tensor_sample_dim(self):
        idx = random.randint(0, 9)
        randitem, _ = self.__getitem__(idx)
        return randitem.size()

    @property
    def padding_used(self):
        return self.pads


def main():
    dataset = ModelsDataset(
        root_dir="/Users/kevingolan/Documents/Coding_Assignments/Thesis/datasets/model_dataset_MNIST/model_dataset.json",
        model_labels_path="/Users/kevingolan/Documents/Coding_Assignments/Thesis/datasets/model_dataset_MNIST/model_dataset.json",
        base_model=MLP(784, 10),
        manipulations=nn_to_2d_tensor,
        padding=True,
    )

    train_dataloader = DataLoader(dataset, 2, True)
    for mbatch_x, mbatch_y in train_dataloader:
        original = dataset.restore_original_tensor(mbatch_x)


if __name__ == "__main__":
    main()

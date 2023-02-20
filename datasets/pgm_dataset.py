import torch
import torch.nn as nn
import json
import copy
import pyro
from utils.graphical_model_utils import layer_wise_dataset
from models.mlp import MLP


class PGMDataset(object):
    def __init__(self, model_json_path: str, base_model: nn.Module):
        self.model_json_path = model_json_path
        self.base_model = base_model
        with open(self.model_json_path) as file:
            self.models_dict: dict = json.load(file)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Converts list of NN objects to a tuple of lists containing layerwise datasets of weights and biases.
        """
        weights, biases = layer_wise_dataset(self._model_list())
        return weights, biases

    def _model_list(self):
        """
        Creates list of NN objects
        """
        model_list = []
        for path in self.models_dict.keys():
            model_list.append(self.load_model(path))
        return model_list

    def load_model(self, model_path):
        """
        Loads state dict of trained model specified by model_path into nn.Module object
        """
        model: nn.Module = copy.deepcopy(self.base_model)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

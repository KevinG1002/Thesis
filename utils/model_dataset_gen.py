import torch
import copy
import json
import torch.nn as nn
import sys
import os
from params import *
from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST, CIFAR10
from torch.optim import Adam

sys.path.append("../")
from experiments.basic_training import run_basic_training, CONFIG
from models.mlp import MLP
from frameworks.sgd_template import SupervisedLearning


class GENCONFIG:
    def __init__(self, num_runs: int = 300, run_config=None, target_dir: str = None):
        self.num_runs = num_runs
        self.run_config = run_config
        self.target_dir = target_dir
        assert run_config, "Need to provide run configuration."
        assert target_dir, "Target directory for trained models needs to be provided."


def reset_parameters(model: nn.Module):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

    return model


def run(cfg: GENCONFIG):
    if not os.path.exists(os.path.join(cfg.target_dir, "models/")):
        os.mkdir(f"{cfg.target_dir}/models/")

    dataset_dict = {}
    models = [MLP() for _ in range(cfg.num_runs)]
    print([[layer_1 for layer_1 in models[i].parameters()][0] for i in range(4)])
    for i in range(cfg.num_runs):
        cfg.run_config.model = models[i]
        cfg.run_config.optim = Adam(
            models[i].parameters(),
            cfg.run_config.learning_rate,
            weight_decay=cfg.run_config.weight_decay,
        )

        print(id(cfg.run_config.model))
        model_path = f"{cfg.target_dir}models/mlp_mnist_model_{i}.pth"
        trained_model, performance_dict = run_basic_training(cfg.run_config)
        torch.save(trained_model.state_dict(), model_path)
        dataset_dict[model_path] = performance_dict["test_loss"].numpy()

    return dataset_dict


def main():
    loss_func = CrossEntropyLoss()
    params = argument_parser()
    target_dataset = "MNIST"
    target_directory = f"../datasets/model_dataset_{target_dataset}/"
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)

    run_cfg = CONFIG(
        MLP(),
        params.dataset,
        10,
        params.num_epochs,
        params.batch_size,
        params.learning_rate,
        params.weight_decay,
        None,
        loss_func,
    )

    cfg = GENCONFIG(num_runs=10, run_config=run_cfg, target_dir=target_directory)
    dataset_dict = run(cfg)
    with open("model_dataset.json", "w") as file:
        json.dump(dataset_dict, file)


if __name__ == "__main__":
    main()

import torch, os
import torch.nn as nn
import datetime
from typing import Callable
from models.mlp import MLP
from datasets.get_dataset import DatasetRetriever
from utils.params import argument_parser
from torch.nn import CrossEntropyLoss
from frameworks.snapshot_ensembles import SnapshotEnsemble


class CONFIG:
    def __init__(
        self,
        base_model: nn.Module,
        dataset_name: str,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        M_snapshots: int,
        criterion: Callable,
    ):
        self.base_model = base_model
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.M_snapshots = M_snapshots
        self.criterion = criterion

        # Metadata instance variables
        dataset = DatasetRetriever(self.dataset_name)
        self.train_set, self.test_set = dataset()
        self.experiment_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_SnapshotEnsembles_{M_snapshots}_snapshots_{epochs}_e_{batch_size}_b_{learning_rate}_lr"
        print(os.getcwd())


def run_experiment(cfg: CONFIG):
    snapshot_process = SnapshotEnsemble(
        cfg.base_model,
        cfg.train_set,
        cfg.test_set,
        None,
        10,
        cfg.batch_size,
        cfg.epochs,
        cfg.learning_rate,
        cfg.M_snapshots,
        cfg.criterion,
    )
    snapshot_process.train()
    snapshot_process.test_ensemble()


if __name__ == "__main__":
    base_model = MLP(784, 10)
    dataset_name = "MNIST"
    training_params = argument_parser()
    cfg = CONFIG(
        base_model,
        dataset_name,
        batch_size=training_params.batch_size,
        epochs=training_params.num_epochs,
        learning_rate=training_params.learning_rate,
        M_snapshots=training_params.M_snapshots,
        criterion=CrossEntropyLoss(),
    )
    run_experiment(cfg)

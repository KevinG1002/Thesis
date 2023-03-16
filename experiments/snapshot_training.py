import torch, os
import torch.nn as nn
import datetime
from typing import Callable
from torch.nn import CrossEntropyLoss
from models.mlp import MLP
from datasets.get_dataset import DatasetRetriever
from utils.params import argument_parser
from utils.exp_logging import Logger
from frameworks.snapshot_ensembles import SnapshotEnsemble

EXPERIMENTAL_RESULTS_PATH = "experimental_results"


class CONFIG:
    def __init__(
        self,
        base_model: nn.Module,
        dataset_name: str,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        lr_min: float,
        M_snapshots: int,
        criterion: Callable,
        log_training: bool = False,
        checkpoint_every: int = None,
    ):
        self.base_model = base_model
        self.model_name = self.base_model.name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lr_min = lr_min
        self.M_snapshots = M_snapshots
        self.criterion = criterion
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Metadata instance variables
        dataset = DatasetRetriever(self.dataset_name)
        self.train_set, self.test_set = dataset()
        self.experiment_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_SnapshotEnsembles_{M_snapshots}_snapshots_{epochs}_e_{batch_size}_b_{learning_rate}_lr"
        self.log_training = log_training
        self.experiment_config = {
            k: v
            for k, v in self.__dict__.items()
            if type(v) in [str, int, float, bool, tuple, list]
        }
        print("\nExperiment Config:\n%s" % self.experiment_config)
        if self.log_training:
            self.checkpoint_every = checkpoint_every
            if os.path.isdir(EXPERIMENTAL_RESULTS_PATH):
                self.experiment_dir = os.path.join(
                    EXPERIMENTAL_RESULTS_PATH, self.experiment_name
                )
                if not os.path.exists(self.experiment_dir):
                    os.mkdir(self.experiment_dir)

                self.logger = Logger(
                    self.experiment_name, self.experiment_dir, self.experiment_config
                )
            else:
                os.mkdir(EXPERIMENTAL_RESULTS_PATH)
                self.experiment_dir = os.path.join(
                    EXPERIMENTAL_RESULTS_PATH, self.experiment_name
                )
                if not os.path.exists(self.experiment_dir):
                    os.mkdir(self.experiment_dir)

                self.logger = Logger(
                    self.experiment_name, self.experiment_dir, self.experiment_config
                )
            self.checkpoint_path = (
                self.logger.checkpoint_path
            )  # automatically generated with logger object.
        else:
            self.checkpoint_path = None
            self.checkpoint_every = None


def run_experiment(cfg: CONFIG):
    snapshot_process = SnapshotEnsemble(
        cfg.base_model.to(cfg.device),
        cfg.train_set,
        cfg.test_set,
        None,
        10,
        cfg.batch_size,
        cfg.epochs,
        cfg.learning_rate,
        cfg.lr_min,
        cfg.M_snapshots,
        cfg.criterion,
        cfg.device,
        cfg.checkpoint_every,
        cfg.checkpoint_path,
    )
    print("\nStart of M-Snapshot Ensemble Experiment")
    print("\nExperiment Name: %s" % cfg.experiment_name)
    print(
        "\nNOTE: if you have the logging flag as True and you intend to checkpoint your models, \nplease consider setting of frequency of checkpoints as frequency of lr_cycles in cyclical learning rate, i.e. every epochs/M_snapshot times."
    )
    train_metrics = snapshot_process.train()
    test_metrics = snapshot_process.test_ensemble()

    if cfg.log_training:
        cfg.logger.save_results(train_metrics, "train_metics.json")
        cfg.logger.save_results(test_metrics, "test_metrics.json")


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
        lr_min=0.0,
        M_snapshots=training_params.M_snapshots,
        criterion=CrossEntropyLoss(),
        log_training=True,
        checkpoint_every=training_params.save_every,
    )
    run_experiment(cfg)

import torch
import datetime
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss
import os
from models.mlp import MLP
from frameworks.sgd_template import SupervisedLearning
from utils.params import argument_parser
from utils.exp_logging import Logger
from datasets.get_dataset import DatasetRetriever

torch.manual_seed(16)

EXPERIMENTAL_RESULTS_PATH = "experimental_results"


class CONFIG:
    def __init__(
        self,
        model: nn.Module,
        dataset_name: str,
        num_classes: int = 10,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.999,
        loss_function: _Loss = None,
        log_training: bool = False,
        checkpoint_every: int = None,
    ):
        self.model = model
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.log_training = log_training

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_function = loss_function if loss_function else CrossEntropyLoss()
        dataset = DatasetRetriever(self.dataset_name)
        self.train_set, self.test_set = dataset()

        self.model_name = self.model.name
        self.experiment_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_SGD_{self.model_name}_{self.dataset_name}_e_{self.epochs}_{self.batch_size}_b_{self.learning_rate}_lr"
        self.experiment_config = {
            k: v
            for k, v in self.__dict__.items()
            if type(v) in [str, int, float, bool, tuple, list]
        }
        print("\nExperiment Config:\n%s" % self.experiment_config)
        # Checkpointing attributes; if log_training set to false, path stay at None and checkpoint every does't change.
        self.checkpoint_path = None
        self.checkpoint_every = checkpoint_every
        if self.log_training:
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


def run_basic_training(cfg: CONFIG):
    supervised_learning_experiment = SupervisedLearning(
        model=cfg.model.to(cfg.device),
        train_set=cfg.train_set,
        test_set=cfg.test_set,
        val_set=None,
        num_classes=cfg.num_classes,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        criterion=cfg.loss_function,
        device=cfg.device,
        checkpoint_every=cfg.checkpoint_every,
        checkpoint_dir=cfg.checkpoint_path,
    )
    train_metrics = supervised_learning_experiment.train()
    test_metrics = supervised_learning_experiment.test()
    if cfg.log_training:
        cfg.logger.save_results(train_metrics)
        cfg.logger.save_results(test_metrics)


def main():
    model = MLP()
    params = argument_parser()
    cfg = CONFIG(
        model,
        "MNIST",
        10,
        params.num_epochs,
        params.batch_size,
        params.learning_rate,
        params.weight_decay,
        log_training=True,
        checkpoint_every=params.save_every,
    )
    run_basic_training(cfg)


if __name__ == "__main__":
    main()

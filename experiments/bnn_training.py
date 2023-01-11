import sys
import torch
import os
import datetime
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss
import torch.nn as nn
from torch.distributions import Uniform
from frameworks.vi_template import VITemplate
from models.bnn import SimpleBNN
from distributions.gaussians import *
from distributions.laplace import LaPlaceDistribution
from distributions.uniform import UniformDistribution
from datasets.get_dataset import DatasetRetriever
from utils.params import argument_parser
from utils.logging import Logger

EXPERIMENTAL_RESULTS_PATH = "experimental_results"


class CONFIG:
    def __init__(
        self,
        model: nn.Module,
        dataset_name: str,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        num_mc_samples: int = 25,
        loss_function: _Loss = None,
        prior_dist: ParameterDistribution = None,
        log_training: bool = True,
        checkpoint_every: int = None,
    ):
        self.model = model
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_mc_samples = num_mc_samples
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prior_dist = prior_dist if prior_dist else UnivariateGaussian(0, 1)
        self.loss_function = loss_function if loss_function else CrossEntropyLoss()

        # Extra attributes for metadata
        self.log_training = log_training
        self.prior_name = self.prior_dist.name
        self.model_name = self.model.name
        self.experiment_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.model_name}_{self.dataset_name}_e_{self.epochs}_prior_{self.prior_dist.name}_{self.num_mc_samples}_mc_samples"
        self.experiment_config = {
            k: v
            for k, v in self.__dict__.items()
            if type(v) in [str, int, float, bool, tuple, list]
        }
        print("\nExperiment Config:\n%s" % self.experiment_config)
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = None
        dataset = DatasetRetriever(self.dataset_name)
        self.train_set, self.test_set = dataset()
        if self.log_training:
            if os.path.isdir(EXPERIMENTAL_RESULTS_PATH):
                experiment_dir = os.path.join(
                    EXPERIMENTAL_RESULTS_PATH, self.experiment_name
                )
                if not os.path.exists(experiment_dir):
                    os.mkdir(experiment_dir)

                self.logger = Logger(
                    self.experiment_name, experiment_dir, self.experiment_config
                )
            else:
                os.mkdir(EXPERIMENTAL_RESULTS_PATH)
                experiment_dir = os.path.join(
                    EXPERIMENTAL_RESULTS_PATH, self.experiment_name
                )
                if not os.path.exists(experiment_dir):
                    os.mkdir(experiment_dir)

                self.logger = Logger(
                    self.experiment_name, experiment_dir, self.experiment_config
                )
            self.checkpoint_path = (
                self.logger.checkpoint_path
            )  # automatically generated with logger object.


def run_bnn_training(cfg: CONFIG):

    vi_experiment = VITemplate(
        model=cfg.model.to(cfg.device),
        train_set=cfg.train_set,
        test_set=cfg.test_set,
        val_set=None,
        num_classes=len(cfg.train_set.classes),
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        num_mc_samples=cfg.num_mc_samples,
        epochs=cfg.epochs,
        likelihood_criterion=cfg.loss_function,
        device=cfg.device,
        checkpoint_every=cfg.checkpoint_every,
        checkpoint_dir=cfg.checkpoint_path,
    )

    train_metrics = vi_experiment.train()
    test_metrics = vi_experiment.evaluate()
    if cfg.log_training:
        cfg.logger.save_results(train_metrics)
        cfg.logger.save_results(test_metrics)


def main():
    bnn = SimpleBNN(
        784,
        10,
        UnivariateGaussian(0, 0.5),
    )
    params = argument_parser()

    cfg = CONFIG(
        model=bnn,
        dataset_name="MNIST",
        epochs=params.num_epochs,
        batch_size=params.batch_size,
        learning_rate=params.learning_rate,
        log_training=True,
        checkpoint_every=params.save_every,
    )
    run_bnn_training(cfg)


if __name__ == "__main__":
    main()
    # print(torch.log(torch.exp(torch.tensor(-2))))

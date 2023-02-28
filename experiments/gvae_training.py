import torch
from typing import Callable
import torch.nn as nn
import torch_geometric as PyG
import torch.nn.functional as F
from torch_geometric.data import Dataset
from torch.optim import Adam
from datasets.graph_dataset import GraphDataset
from frameworks.sgd_template import SupervisedLearning
from frameworks.gvae_template import GVAE_Training
from utils.exp_logging import checkpoint
from models.graph_vae import GraphVAE, Encoder, Decoder, GVAELoss
from models.mlp import MLP
from utils.exp_logging import Logger
import os
import datetime

EXPERIMENTAL_RESULTS_PATH = "experimental_results"


class CONFIG:
    def __init__(
        self,
        gvae: GraphVAE,
        dataset: Dataset,
        criterion: Callable,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        decay_rate: float,
        num_samples: int,
        log_training: bool = False,
        base_model: nn.Module = MLP(),
        checkpoint_path: str = None,
        checkpoint_every: int = None,
    ):
        self.gvae = gvae
        self.base_model = base_model
        self.dataset = dataset
        self.criterion = criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.num_samples = num_samples
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = self.gvae.name
        self.experiment_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.model_name}_GraphDataset_e_{self.epochs}_{self.learning_rate}_lr_{self.batch_size}_b"
        self.log_training = log_training
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self.experiment_config = {
            k: v
            for k, v in self.__dict__.items()
            if type(v) in [str, int, float, bool, tuple, list]
        }
        print("\nExperiment Config:\n%s" % self.experiment_config)
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


def run(cfg: CONFIG):
    GVAE_training_process = GVAE_Training(
        cfg.gvae,
        cfg.base_model,
        cfg.dataset,
        cfg.criterion,
        cfg.epochs,
        cfg.batch_size,
        cfg.learning_rate,
    )
    GVAE_training_process.train()


if __name__ == "__main__":
    gvae_enc = Encoder(1)
    gvae_dec = Decoder(1)
    loss_fn = GVAELoss()

    dataset = GraphDataset(
        base_model=MLP(),
        root="../datasets/model_dataset_MNIST",
    )
    graph_vae = GraphVAE(gvae_enc, gvae_dec)
    cfg = CONFIG(
        graph_vae,
        dataset,
        loss_fn,
        1,
        10,
        1e-3,
        0.0,
        10,
    )
    run(cfg)
import torch
from typing import Callable
import torch.nn as nn
import torch_geometric as PyG
import torch.nn.functional as F
from torch_geometric.data import Dataset, Batch
from torch.optim import Adam
from frameworks.sgd_template import SupervisedLearning
from utils.exp_logging import checkpoint
from models.graph_vae import GraphVAE, Encoder, Decoder, GVAELoss
from torch_geometric.loader import DataLoader, NeighborSampler, NeighborLoader
from samplers.graph_sampler import GVAE_Sampler

import os


class GVAE_Training:
    def __init__(
        self,
        gvae: GraphVAE,
        base_model: nn.Module,
        dataset: Dataset,
        criterion: Callable,
        epochs: int = 50,
        batch_size: int = 5,
        learning_rate: float = 1e-3,
        decay_rate: float = 0,
        num_samples: int = 10,
        log_training: bool = False,
        checkpoint_every: int = None,
        subgraph_sampling: bool = True,
        device: str = "cpu",
    ):

        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.criterion: GVAELoss = criterion
        self.model: GraphVAE = gvae  # for typing purposes
        self.base_model = base_model
        self.dataset = dataset
        self.optimizer = Adam(
            self.model.parameters(), learning_rate, weight_decay=decay_rate
        )
        self.log_training = log_training
        self.num_samples = num_samples
        self.graph_edge_index = (
            self.dataset.default_edge_index
        )  # in our problem, all graphs have the exact same edge_index since they all have the same connections but different weights (node attributes here)
        print(self.graph_edge_index.size())
        if subgraph_sampling:
            self.subgraph_sampling = subgraph_sampling  # GVAE_Sampler("uniform", 2.5, self.graph_edge_index, 100)

        self.device = device
        self.checkpoint_every = checkpoint_every

    def _instantiate_train_dataloader(self):
        return DataLoader(self.dataset, self.batch_size, shuffle=True)

    def train(self) -> dict:
        self.model.train(True)
        if not self.subgraph_sampling:
            self.train_dataloader = self._instantiate_train_dataloader()
        else:
            self.train_dataloader = NeighborLoader(
                Batch.from_data_list(self.dataset),
                [100, 100, 100],
                batch_size=10000,
            )
        print("\n\n")
        for epoch in range(self.epochs):
            print("\nStart of training epoch %d." % (epoch + 1))
            train_ep_metrics = self.train_epoch(epoch + 1)
            print("\nTraining for epoch %d done." % (epoch + 1))
            if epoch < 1:
                train_metrics = {
                    k: [v] for k, v in train_ep_metrics.items()
                }  # create epoch-wise dict of training metrics.
            else:
                for k in train_ep_metrics.keys():
                    train_metrics[k].append(train_ep_metrics[k])
            if self.checkpoint_every and ((epoch + 1) % self.checkpoint_every == 0):
                checkpoint_name = "%s_checkpoint_e_%d_loss_%.3f.pt" % (
                    self.model.name,
                    epoch + 1,
                    train_ep_metrics["train_loss"],
                )
                model_checkpoint_path = os.path.join(
                    self.checkpoint_dir, checkpoint_name
                )
                checkpoint(
                    model_checkpoint_path,
                    epoch + 1,
                    self.model.state_dict(),
                    self.optimizer.state_dict(),
                    train_metrics["train_loss"],
                )
        if self.checkpoint_every:
            checkpoint_name = "%s_fully_trained_e_%d_loss_%.3f.pt" % (
                self.model.name,
                self.epochs,
                train_ep_metrics["train_loss"],
            )
            model_checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            checkpoint(
                model_checkpoint_path,
                epoch + 1,
                self.model.state_dict(),
                self.optimizer.state_dict(),
                train_metrics["train_loss"],
            )
        return train_metrics

    def train_epoch(self, epoch_idx: int):
        train_loss = 0.0
        edge_index = self.graph_edge_index
        num_iterations = 100
        # for _ in range(num_iterations):
        for idx, mbatch_x in enumerate(self.train_dataloader):
            # mbatch_x = next(iter(self.train_dataloader))
            self.optimizer.zero_grad()
            print(mbatch_x)
            print(mbatch_x.edge_index.size())
            print(mbatch_x.x.size())
            print(mbatch_x.x)
            print(
                mbatch_x.x == mbatch_x.edge_weight
            )  # need to apply transform that removes edge weight
            mbatch_x = mbatch_x.to(self.device)
            x = mbatch_x.x
            edge_index = mbatch_x.edge_index
            num_nodes = mbatch_x.num_nodes
            # if self.subgraph_sampling:
            # x, edge_index = self.sampler(x, edge_index)
            recon_z, mu, log_var = self.model(x, edge_index, None)

            loss = self.criterion(
                x,
                recon_z,
                num_nodes,
                log_var,
                mu,
            )
            # if idx % 100 == 0:
            # print("ELBO loss:", loss.item())
            loss.backward()
            train_loss += loss
            self.optimizer.step()
            print("ELBO loss:", loss.item())

        avg_train_loss = train_loss.item() / len(self.train_dataloader)

        return {
            "train_loss": avg_train_loss,
        }

    def sample(self):
        mu = self.model.mu
        log_var = self.model.log_var
        gen_graphs = []
        for _ in range(self.num_samples):
            z = self.model.reparametrize(mu, log_var)
            gen_graphs.append(
                self.model.decoder(
                    z,
                    self.graph_edge_index,
                )
            )
        return gen_graphs

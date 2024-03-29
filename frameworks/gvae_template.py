import torch
from typing import Callable
import numpy as np
import torch.nn as nn
import torch_geometric as PyG
import torch.nn.functional as F
from torch_geometric.data import Dataset, Batch, Data
from torch.optim import Adam
from frameworks.sgd_template import SupervisedLearning
from utils.exp_logging import checkpoint, Logger
from models.graph_vae import GraphVAE, Encoder, Decoder, GVAELoss
from torch_geometric.loader import DataLoader, NeighborSampler, NeighborLoader
from samplers.graph_sampler import GVAE_Sampler
from utils.graph_manipulations import pygeometric_to_nn
from datasets.get_dataset import DatasetRetriever
from torch_scatter import scatter
import copy

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
        grad_accumulation: int = 5,
        log_training: bool = False,
        checkpoint_every: int = None,
        subgraph_sampling: bool = False,
        device: str = "cpu",
        logger: Logger = None,
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
        self.grad_accumulation = grad_accumulation
        self.graph_edge_index = (
            self.dataset.default_edge_index
        )  # in our problem, all graphs have the exact same edge_index since they all have the same connections but different weights (node attributes here)
        # print(self.graph_edge_index.size())

        # GVAE_Sampler("uniform", 2.5, self.graph_edge_index, 100)
        self.subgraph_sampling = subgraph_sampling

        self.device = device
        self.checkpoint_every = checkpoint_every
        self.logger = logger

        # For the evaluation of sampled graphs.
        gen_model_test_dataset = DatasetRetriever("MNIST")
        _, self.test_set = gen_model_test_dataset()

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
            train_ep_metrics, sample_model_metrics = self.train_epoch(
                epoch + 1)
            print("\nTraining for epoch %d done." % (epoch + 1))
            if self.logger:
                self.logger.save_results(
                    sample_model_metrics, f"sample_model_metrics_epoch_{epoch+1}.json")
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
            model_checkpoint_path = os.path.join(
                self.checkpoint_dir, checkpoint_name)
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
            # print(
            #     mbatch_x.x == mbatch_x.edge_weight
            # )  # need to apply transform that removes edge weight
            mbatch_x = mbatch_x.to(self.device)
            # x = mbatch_x.edge_attr
            edge_index = mbatch_x.edge_index
            x = mbatch_x.x
            num_nodes = mbatch_x.num_nodes
            # if self.subgraph_sampling:
            # x, edge_index = self.sampler(x, edge_index)
            recon_z, self.mu, self.log_var = self.model(x, edge_index, None)

            loss = (
                self.criterion(
                    x,
                    recon_z,
                    num_nodes,
                    self.log_var,
                    self.mu,
                )
                / self.grad_accumulation
            )
            if idx % 30 == 0:
                print("ELBO loss:", loss.item())
            loss.backward()
            train_loss += loss * self.grad_accumulation
            if (idx + 1) % self.grad_accumulation:
                self.optimizer.step()
                self.optimizer.zero_grad()
            # print("ELBO loss:", loss.item())

        avg_train_loss = train_loss.item() / len(self.train_dataloader)
        eval_metrics, eval_avg_metrics = self.eval_epoch(epoch_idx)
        return {
            "train_loss": avg_train_loss,
        }.update(eval_avg_metrics), eval_metrics

    @torch.no_grad()
    def sample_graphs(self) -> list:
        # mu = self.model.mu
        # log_var = self.model.log_var
        gen_graphs = []
        self.model.eval()
        for _ in range(self.num_samples):
            z = self.model.reparametrize(self.mu, self.log_var) + torch.randn_like(
                self.log_var
            ) * torch.exp(self.log_var)
            g = self.model.decode(z, self.graph_edge_index, None)
            g = g.view(
                int(g.size()[0] / self.dataset.default_number_of_nodes), -1)
            # print(g)
            new_g = Data(edge_index=self.graph_edge_index, x=g[0], weight=g[0])
            gen_graphs.append(new_g)
        return gen_graphs

    @torch.no_grad()
    def eval_samples(self, generated_graphs):
        eval_results = {
            "test_loss": [],
            "test_acc": [],
            "f1_metric": [],
            "recall": [],
            "precision": [],
            "distinct_count": [],
        }
        for idx, g in enumerate(generated_graphs):
            nn = pygeometric_to_nn(copy.deepcopy(g), self.base_model)
            test_process = SupervisedLearning(
                nn, test_set=self.test_set, device=self.device
            )
            print("\nTesting Generated Sample %d:\n" % (idx + 1))

            sample_test_metrics = test_process.test()
            for key in eval_results.keys():
                eval_results[key].append(sample_test_metrics[key])
        eval_avg_result = {}
        for k, v in eval_results.items():
            if type(v[0]) != list:
                eval_avg_result[f"mean_{k}"] = sum(v)/len(v)
            else:
                eval_avg_result[f"mean_{k}"] = np.mean(
                    np.array(v), axis=0).tolist()
        return eval_results, eval_avg_result

    def eval_epoch(self, idx):
        print("\nEvaluating graph samples -- Epoch %d" % idx)
        sampled_graphs = self.sample_graphs()
        return self.eval_samples(sampled_graphs)

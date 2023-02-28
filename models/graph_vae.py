import torch_geometric as PyG
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GCN, VGAE, ARGVA
from torch_geometric.nn import GCNConv
from datasets.graph_dataset import GraphDataset
from models.mlp import MLP
from torch_geometric.transforms import LineGraph
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from models.mlp import MLP
from utils.graph_manipulations import weight_tensor_to_graph
from utils.profile import profile


class GVAELoss(nn.Module):
    def __init__(self):
        super(GVAELoss, self).__init__()

    def forward(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        num_nodes: int,
        log_var: torch.Tensor,
        mu: torch.Tensor,
        reduction_fn: str = "sum",
    ):
        loss = F.binary_cross_entropy_with_logits(
            predictions, ground_truth, reduction=reduction_fn
        )  # log-likelihood

        kl = (0.5 / num_nodes) * torch.mean(
            torch.sum(
                1 + 2 * log_var - torch.square(mu) - torch.square(torch.exp(log_var)),
                1,
            )
        )
        loss -= kl
        return loss


class Encoder(nn.Module):
    """
    Encoder for VGAE. Pretty simple, used ChatGPT for help.
    """

    def __init__(self, dataset_in_channels):
        super(Encoder, self).__init__()
        self.conv_1 = GCNConv(dataset_in_channels, 16)
        self.conv_2 = GCNConv(16, 8)
        self.conv_mu = GCNConv(
            8, 2
        )  # to parametrize latent space gaussian distribution
        self.conv_logvar = GCNConv(
            8, 2
        )  # to parametrize latent space gaussian distribution

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv_1(x, edge_index, edge_weight))
        x = F.relu(self.conv_2(x, edge_index, edge_weight))
        mu = self.conv_mu(x, edge_index, edge_weight)
        log_var = self.conv_logvar(x, edge_index, edge_weight)
        return mu, log_var


class Decoder(nn.Module):
    """
    Decoder for VGAE. Pretty simple, used ChatGPT for help.
    """

    def __init__(self, dataset_out_channels):
        super(Decoder, self).__init__()
        self.upconv1 = GCNConv(2, 8)
        self.upconv2 = GCNConv(8, 16)
        self.upconv3 = GCNConv(16, dataset_out_channels)

    def forward(self, z, edge_index, edge_weight):
        z = F.relu(self.upconv1(z, edge_index, edge_weight))
        z = F.relu(self.upconv2(z, edge_index, edge_weight))
        z = self.upconv3(z, edge_index, edge_weight)
        return z


class GraphVAE(VGAE):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__(encoder=encoder, decoder=decoder)
        self.name = self._get_name()
        self.mu = None
        self.log_var = None

    def forward(self, x, edge_index, edge_weight):
        mu, log_var = self.encoder(x, edge_index, edge_weight)
        z = self.reparametrize(mu, log_var)
        restored_x = self.decoder(z, edge_index, edge_weight)
        return restored_x, mu, log_var


def run():
    enc = Encoder(1)
    dec = Decoder(1)
    graph_vae = GraphVAE(enc, dec)
    transforms = T.Compose([weight_tensor_to_graph, LineGraph(True)])

    gd = GraphDataset(
        base_model=MLP(),
        root="../datasets/model_dataset_MNIST",
        pre_transform=transforms,
    )
    print([(gd[0].edge_index == gd[i].edge_index).all() for i in range(4)])
    graph_dataloader = DataLoader(gd, batch_size=4, shuffle=True)
    print(len(graph_dataloader.dataset))
    for batch in graph_dataloader:
        print(batch)
        print(batch.x.size())


if __name__ == "__main__":
    run()

import torch_geometric as PyG
import torch
import torch.nn as nn
from torch_geometric.nn.models import GCN, VGAE, ARGVA
from torch_geometric.nn import GCNConv
from datasets.graph_dataset import GraphDataset
from models.mlp import MLP
from torch_geometric.transforms import LineGraph
import torch_geometric.transforms as T
from models.mlp import MLP
from utils.graph_manipulations import weight_tensor_to_graph


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
        x = nn.ReLU(self.conv_1(x, edge_index, edge_weight))
        x = nn.ReLU(self.conv_2(x, edge_index, edge_weight))
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
        z = nn.ReLU(self.upconv1(z, edge_index, edge_weight))
        z = nn.ReLU(self.upconv2(z, edge_index, edge_weight))
        z = self.upconv3(z, edge_index, edge_weight)
        return z


def run():
    enc = Encoder(1)
    dec = Decoder(1)
    graph_vae = VGAE(enc, dec)
    transforms = T.Compose([weight_tensor_to_graph, LineGraph(True)])

    gd = GraphDataset(
        base_model=MLP(),
        root="../datasets/model_dataset_MNIST",
        pre_transform=transforms,
    )
    graph_vae.eval()
    graph_vae(gd[0])


if __name__ == "__main__":
    run()

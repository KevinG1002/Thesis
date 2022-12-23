# UNet script based off of https://nn.labml.ai/diffusion/ddpm/unet.html; full credit to them.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    pass


class TimeEmbedding(nn.Module):
    def __init__(self, nchannels: int):
        """
        TimeEmbedding layer class.
        Args:
            - nchannels: number of dimensions in the embedding.
        """
        super().__init__()
        self.nchannels = nchannels

        self.fc1 = nn.Linear(self.nchannels // 4, self.nchannels)  # First linear layer
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(self.nchannels, self.nchannels)  # Second linear layer

    def forward(self, t: torch.Tensor):
        """
        The forward function of the time embedding creates a sinusoidal position embedding for the time step tensor
        and then transforms it with an MLP.
        Args:
            - t: time step tensor
        """
        half_dim = self.nchannels // 8
        embedding = math.log(10000) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=t.device) * -embedding)
        embedding = t[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=1)

        embedding = self.activation(self.fc1(embedding))
        embedding = self.fc1(embedding)
        return embedding


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_channels: int, n_groups: int
    ):
        """
        ResidualBlock class for wider UNet implementation.
        Args:
            - in_channels: number of input channels
            - out_channels: number of output channels
            - time_channels: dimensions in diffusion timestep embedding
            - n_groups: number of groups for group normalization (alternative approach to normalization within a batch of samples)
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.activation_1 = nn.SiLU()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.activation_2 = nn.SiLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # If number of output channels is not equal to the number of input channels, the shortcut connection needs to be projected to a higher dimension.
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        self.time_embedding = nn.Linear(
            time_channels, out_channels
        )  # Linear layer for time embeddings

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: input sample with shape [batch_size, in_channels, heigh, width]
        t: diffusion time step embedding sample with shape [batch_size, time_channels]
        """
        h = self.conv1(
            self.activation_1(self.norm1(x))
        )  # First conv layer with group normalization
        h += self.time_embedding(t)[
            :, :, :None, None
        ]  # Add time embedding to output of first conv_layer
        h = self.conv2(
            self.activation_2(self.norm2(h))
        )  # Second conv layer with group normalization

        return h + self.shortcut(x)  # Add skip/shortcut connection to output


class AttentionBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_heads: int = 1,
        d_k: int = None,
        n_groups: int = 32,
    ):
        """
        Attention Block for our UNet bottleneck.
        Args:
            - n_channels: number of channels in the input
            - n_heads: number of heads in multihead attention
            - d_k: dimensions in each head
            - n_groups: number of groups for group normalization
        """
        super().__init__()
        if not d_k:
            d_k = n_channels

        self.norm = nn.GroupNorm(n_groups, n_channels)  # Normalization layer
        self.projection = nn.Linear(
            n_channels, n_heads * d_k * 3
        )  # Projection for query, key and values
        self.output = nn.Linear(
            n_heads * d_k, n_channels
        )  # Linear Layer for final transformation.

        self.scale = d_k ** (-0.5)  # Scale for dot-product attention
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: torch.Tensor = None):
        """
        Args:
            - x: input sample, has shape [batch_size, n_channels, height, width]
            - t: input diffusion steps, has shape [batch_size, time_channels]
        """
        if not t:
            _ = t

        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(
            0, 2, 1
        )  # x is first viewed as batch_size, n_channels and then a sequence flattening height and width.
        # with permute, we obtain the following shape: batch_size, sequence, and then channels.
        qkv = self.projection(x).view(
            batch_size, -1, self.n_heads, 3 * self.d_k
        )  # Gets query, key, values in a concatenated representation, and then shapes it to [batch_size, seq, n_heads, 3*d_k]

        q, k, v = torch.chunk(
            qkv, 3, dim=-1
        )  # splits qkv concatenated tensor into three separate tensors - query, key and values. Each of them will have shape: [batch_size, seq, n_heads, d_k]

        attention = (
            torch.einsum("bihd, bjhd->bijh", q, k) * self.scale
        )  # Calculates scaled dot-product of query and keys.

        attention = F.softmax(attention, dim=2)

        # Multiply by the value tensor
        res = torch.einsum("bijh,bjhd->bihd", attention, v)

        # Reshape results to have shape: [batch_size, sequence, n_heads * d_k]
        res = res.view(batch_size, -1, self.n_heads * self.d_k)

        # Transform to shape: [batch_size, sequence, n_channels]
        res: torch.Tensor = self.output(res)

        res += x  # add skip connection

        # Change to shape [batch_size, in_channels, height, width]
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attention: bool,
    ) -> None:
        """
        This class combines a residual block and an attention block. We use a Downblock on the first half of the UNet architecture (on the left handside when we go down in resolution)
        The residual block usually precedes the attention block. So the input of the attention block should be the number of channels at the output of the residual block (i.e. out channels)
        Output shape of Downblock: [batch_size, out_channels, height, width]
        Args:
            - in_channels: number of channels in sample inputted in block.
            - out_channels: number of desired output channels for our downsampling scheme as we go across the DownBlock
            - time_channels: number of channels for our diffusion time step embeddings
            - has_attention: boolean denoting whether or not our DownBlock uses attention.
        """
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x: batch of input samples with shape: [batch_size, n_channels, height, width]
            t: batch of timestep embeddings with shape: [batch_size, time_channels]
        """
        x = self.res(x, t)
        x = self.attention(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attention: bool,
    ) -> None:
        """
        Combines a residual block with an attention block in the second half of the UNet where upsampling
        takes place. Here, the input of the residual block has in_channels = (in_channels + out_channels) whereas
        the out_channels of the residual block = out_channels. The attention block, if included, takes out_channels as an input
        and produces an output with shape: [batch_size, out_channels, height, width]

        """
        super().__init__()
        self.res = ResidualBlock(
            in_channels + out_channels, out_channels, time_channels
        )
        if has_attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attention(x)
        return x


class DDPMUNet(nn.Module):
    def __init__(self):
        pass

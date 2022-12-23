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
        """
        Args:
            x: batch of input samples with shape: [batch_size, n_channels, height, width]
            t: batch of timestep embeddings with shape: [batch_size, time_channels]
        """
        x = self.res(x, t)
        x = self.attention(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        time_channels: int,
    ):
        """
        The MiddleBlock component of the Unet resides in the middle of the architecture, i.e
        at its bottleneck where the resolution is the lowest. The MiddleBlock is composed of three components: a ResidualBlock, an
        AttentionBlock followed by another ResidualBlock. Here, no contraction or expansion is made in terms of the
        resolution nor depth of our samples, i.e., in_channels = out_channels = n_channels throughout.
        Args:
            - n_channels: The number of channels in input to MiddleBlock.
            - time_channels: The number of time channels in input to MiddleBlock.
        """
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attention = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x: batch of input samples with shape: [batch_size, n_channels, height, width]
            t: batch of timestep embeddings with shape: [batch_size, time_channels]
        """
        x = self.res1(x, t)
        x = self.attention(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    def __init__(self, n_channels: int):
        """
        Upsample block scales-up the resolution by a factor of two.
        For example, if resolution is (4 x 4), this block will upscale it to (8x8)
        Args:
            - n_channels: number of channels in the input.
        """
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            n_channels, n_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        t not used but preserved to match function signature with other blocks
        Args:
            x: batch of input samples with shape: [batch_size, n_channels, height, width]
            t: batch of timestep embeddings with shape: [batch_size, time_channels]
        """
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_channels: int) -> None:
        """
        Downsample block scales the resolution of the input by a factor of 1/2.
        So, if the input image has resolution (4x4), for instance, the output after the downsample block
        will be (2x2).
        Args:
            - n_channels: number of channels in the input.
        """
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        t not used but preserved to match function signature with other blocks
        Args:
            x: batch of input samples with shape: [batch_size, n_channels, height, width]
            t: batch of timestep embeddings with shape: [batch_size, time_channels]
        """
        _ = t
        return self.conv(x)


class DDPMUNet(nn.Module):
    def __init__(
        self,
        sample_channels: int = 1,
        n_channels: int = 64,
        ch_dims: list[int] = [1, 2, 2, 4],
        is_attention: list[bool] = [False, False, True, True],
        n_blocks: int = 1,
    ):
        """
        UNet used to predict addednoise in diffusion process. This UNet is used in combination with the DDPM.
        Args:
            - sample_channels: number of channels in the input sample. 1 for 2D matrix representing weights.
            - n_channels: number of channels in the very first feature map we transform our input samples to.
            - ch_dims: is a list that denotes the channel numbers at each resolution. Number of channels at given
            at i-th resolution is calculated as: ch_dims[i] * n_channels.
            - is_attention: list of booleans denoting if attention should be used in a given resolution.
            - n_blocks: denotes the number of Up & Down blocks at each resolution; Symmetry is preserved in this architecture,
            so number of up blocks will be equal to the number of down blocks.
        """
        super().__init__()
        n_resolutions = len(ch_dims)  # number of resolutions
        self.sample_projection = nn.Conv2d(
            sample_channels, n_channels, kernel_size=(3, 3), padding=(1, 1)
        )  # Projection of sample into feature map with n_channels.
        self.time_embedding = TimeEmbedding(n_channels * 4)  # Time Embedding Layer

        ## First Half of UNet: Decreasing the resolution ##
        down = []

        # Number of channels
        out_channels = in_channels = n_channels

        # For each resolution do:
        for i in range(n_resolutions):
            out_channels = (
                n_channels * ch_dims[i]
            )  # number of channels in given resolution
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels, out_channels, n_channels * 4, is_attention[i]
                    )
                )
            if (
                i < n_resolutions - 1
            ):  # Downsample for each resolution appart from the last
                down.append(Downsample(in_channels))

        # Resulting Down structure: [DownBlock_res_1, Downsample_res_1, DownBlock_res_2, Downsample_res_2, ..., DownBlock_res_n-1, Downsample_res_n-1, DownBlock_res_n]

        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, n_channels * 4)

        up = []
        in_channels = out_channels

        for i in reversed(
            range(n_resolutions)
        ):  # will iterate over [len(n_res), len(n_res) -1, ..., 1, 0]
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(in_channels, out_channels, n_channels * 4, is_attention[i])
                )
            out_channels = (
                in_channels // ch_dims[i]
            )  # reduce number of channels for output
            up.append(
                UpBlock(in_channels, out_channels, n_channels * 4, is_attention[i])
            )  # Add a block to reduce the number of channels
            in_channels = out_channels

            if i > 0:  # Upsample signal at all resolutions expect last
                up.append(Upsample(in_channels))

        self.up = nn.ModuleList(up)

        self.norm = nn.GroupNorm(8, n_channels)
        self.activation = nn.SiLU()
        self.final_layer = nn.Conv2d(
            in_channels, sample_channels, kernel_size=(3, 3), padding=(1, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = self.time_embedding(t)
        x = self.sample_projection(x)

        h = [
            x
        ]  # variable that stores outputs of each resolution for eventual skip connection.

        # For each block in down list of modules (first half of UNet), perform a forward pass of the input and then store the output in h_list.
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Perform forward pass on middle-block with output from list of downblocks:
        x = self.middle(x, t)

        # Perform forward pass with output of middle block on modules of second half of UNet (upwards):
        for u in self.up:
            if isinstance(
                u, Upsample
            ):  # If current block is an UpSampling module, perform forward pass witout skip-connections.
                x = u(x, t)
            else:  # Otherwise, Concatenate skip-connection with current input, then do forward pass.
                skip = h.pop()
                x = torch.cat([x, skip], dim=1)
                x = u(x, t)

        return self.final_layer((self.activation(self.norm(x))))

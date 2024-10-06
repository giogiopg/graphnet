"""Classes for performing embedding of input data."""
import torch
import torch.nn as nn
from torch.functional import Tensor
import math

from typing import Optional, Union, List

from pytorch_lightning import LightningModule


class SinusoidalPosEmb(LightningModule):
    """Sinusoidal positional embeddings module.

    This module is from the kaggle competition 2nd place solution (see
    arXiv:2310.15674): It performs what is called Fourier encoding or it's used
    in the Attention is all you need arXiv:1706.03762. It can be seen as a soft
    digitization of the input data
    """

    def __init__(
        self,
        dim: int = 16,
        n_freq: int = 10000,
        scaled: bool = False,
    ):
        """Construct `SinusoidalPosEmb`.

        Args:
            dim: Embedding dimension.
            n_freq: Number of frequencies.
            scaled: Whether or not to scale the output.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim has to be even. Got: {dim}")
        self.scale = (
            nn.Parameter(torch.ones(1) * dim**-0.5) if scaled else 1.0
        )
        self.dim = dim
        self.n_freq = torch.Tensor([n_freq])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        device = x.device
        half_dim = self.dim / 2
        emb = torch.log(self.n_freq.to(device=device)) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb * self.scale


class FourierEncoder(LightningModule):
    """Fourier encoder module.

    This module incorporates sinusoidal positional embeddings and auxiliary
    embeddings to process input sequences and produce meaningful
    representations. The module assumes that the input data is in the format of
    (x, y, z, time, charge, auxiliary), being the first four features
    mandatory.
    """

    def __init__(
        self,
        seq_length: int = 128,
        mlp_dim: Optional[int] = None,
        output_dim: int = 384,
        scaled: bool = False,
        n_features: int = 6,
    ):
        """Construct `FourierEncoder`.

        Args:
            seq_length: Dimensionality of the base sinusoidal positional
                embeddings.
            mlp_dim (Optional): Size of hidden, latent space of MLP. If not
                given, `mlp_dim` is set automatically as multiples of
                `seq_length` (in consistent with the 2nd place solution),
                depending on `n_features`.
            output_dim: Dimension of the output (I.e. number of columns).
            scaled: Whether or not to scale the embeddings.
            n_features: The number of features in the input data.
        """
        super().__init__()

        self.sin_emb = SinusoidalPosEmb(dim=seq_length, scaled=scaled)
        self.aux_emb = nn.Embedding(2, seq_length // 2)
        self.sin_emb2 = SinusoidalPosEmb(dim=seq_length // 2, scaled=scaled)

        if n_features < 4:
            raise ValueError(
                f"At least x, y, z and time of the DOM are required. Got only "
                f"{n_features} features."
            )
        elif n_features >= 6:

            hidden_dim = 6 * seq_length
        else:
            hidden_dim = int((n_features + 0.5) * seq_length)

        if mlp_dim is None:
            mlp_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )

        self.n_features = n_features

    def forward(
        self,
        x: Tensor,
        seq_length: Tensor,
    ) -> Tensor:
        """Forward pass."""
        length = torch.log10(seq_length.to(dtype=x.dtype))
        embeddings = [self.sin_emb(4096 * x[:, :, :3]).flatten(-2)]  # Position

        if self.n_features >= 5:
            embeddings.append(self.sin_emb(1024 * x[:, :, 4]))  # Charge

        embeddings.append(self.sin_emb(4096 * x[:, :, 3]))  # Time

        if self.n_features >= 6:
            embeddings.append(self.aux_emb(x[:, :, 5].long()))  # Auxiliary

        embeddings.append(
            self.sin_emb2(length).unsqueeze(1).expand(-1, max(seq_length), -1)
        )  # Length

        x = torch.cat(embeddings, -1)
        x = self.mlp(x)

        return x


class SpacetimeEncoder(LightningModule):
    """Spacetime encoder module."""

    def __init__(
        self,
        seq_length: int = 32,
    ):
        """Construct `SpacetimeEncoder`.

        This module calculates space-time interval between each pair of events
        and generates sinusoidal positional embeddings to be added to input
        sequences.

        Args:
            seq_length: Dimensionality of the sinusoidal positional embeddings.
        """
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(dim=seq_length)
        self.projection = nn.Linear(seq_length, seq_length)

    def forward(
        self,
        x: Tensor,
        # Lmax: Optional[int] = None,
    ) -> Tensor:
        """Forward pass."""
        pos = x[:, :, :3]
        time = x[:, :, 3]
        spacetime_interval = (pos[:, :, None] - pos[:, None, :]).pow(2).sum(
            -1
        ) - ((time[:, :, None] - time[:, None, :]) * (3e4 / 500 * 3e-1)).pow(2)
        four_distance = torch.sign(spacetime_interval) * torch.sqrt(
            torch.abs(spacetime_interval)
        )
        sin_emb = self.sin_emb(1024 * four_distance.clip(-4, 4))
        rel_attn = self.projection(sin_emb)
        return rel_attn

class FeaturesProcessing(nn.Module):
    """ Process the hits features by passing them through a embedding block. """

    def __init__(
                    self,
                    emb_dims: Union[List, int],
                    n_features: int = 6,
    ):
        """ Pass all the features through a embedding block before feed them to the model.

            Args:
                n_features: The number of features in the input data.
                emb_dims: Dimensionality of the consecutive linear layers.
        """

        super().__init__()

        if isinstance(emb_dims, int):
            emb_dims = [emb_dims]

        self.model_dim = emb_dims[-1]

        module_list = []
        for emb_dim in emb_dims:
            module_list.extend([
                                    nn.LayerNorm(n_features),
                                    nn.Linear(n_features, emb_dim),
                                    nn.GELU()
            ])
            n_features = emb_dim

        self.emb = nn.Sequential(*module_list)


    def forward(self, x):
        return self.emb(x) * math.sqrt(self.model_dim)

class PositionalEncoding(nn.Module):
    """ Sinusodial Position Embedding for continuous variables."""

    def __init__(
                    self,
                    dim: int = 128,
                    seq_length: int = 300,
    ):
        """ Associate an unique representation to each position in a sequence using Sinusoidal Fourier position encoding.

        Args:
            dim: Dimension of the model
            seq_length: Maximun length of the sequence.

    """

        super().__init__()

        pos_emb = torch.zeros(seq_length, dim)
        positions = torch.arange(0, seq_length, dtype = torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, dim, 2).float() * -math.log(10000.0) / dim  )

        pos_emb[:, 0::2] = torch.sin(positions * div_term)
        pos_emb[:, 1::2] = torch.cos(positions * div_term)

        pos_emb = pos_emb.unsqueeze(0) #pos_emb.hape: [1, seq, dim]

        self.register_buffer('pos_emb', pos_emb)

    def forward(self, x):
        # x.shape = [B, seq_len, dim]
        x = x + self.pos_emb[:, :x.shape[1], :]
        return x

class CausalityMask(LightningModule):
    """ Creates a mask to be passed to the attention weights based on the causality of the pulses."""

    def __init__(
        self,
        seq_length: int = 32,
        refractive_index: float = 1.33
    ):
        """Construct `CausalityMask`.

        This module calculates an attention mask based on the causality between
        each pair of pulses in the sequence.

        Args:
            seq_length: Number of pulses in the event.
        """
        super().__init__()
        self.seq_length = seq_length
        self.refractive_index = refractive_index
        self.c = 299792458.0
        self.v = self.c / self.refractive_index

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward pass."""
        pos = x[:, :, :3]
        time = x[:, :, 3]
        spacetime_interval = (pos[:, :, None] - pos[:, None, :]).pow(2).sum(
            -1
        ) - ((time[:, :, None] - time[:, None, :]) * (self.v * 1e-9)).pow(2)
        four_distance = torch.sign(spacetime_interval) * torch.sqrt(
            torch.abs(spacetime_interval)
        )
        return four_distance.clip(-4, 4)
    
class EuclideanMask(LightningModule):
    """ Creates a mask to be passed to the attention weights based on the distance of the pulses."""

    def __init__(
        self,
        seq_length: int = 32,
        max_distance: float = 50.0
    ):
        """Construct `EuclideanDistanceMask`.

        This module calculates an attention mask based on the distance between
        each pair of pulses in the sequence.

        Args:
            seq_length: Number of pulses in the event.
            max_distance: The maximum distance (in meters) between pulses.
        """
        super().__init__()
        self.seq_length = seq_length
        self.max_distance = max_distance

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward pass."""
        diff = x[:, :, None, :] - x[:, None, :, :]
        euclidean_distance = torch.sqrt(torch.sum(diff**2, dim=-1))
        return euclidean_distance.clip(0, self.max_distance)
    
class IdsMask(LightningModule):
    """ Creates a mask to be passed to the attention weights based on the 
        DU, DOM or PMT ids of the pulses."""

    def __init__(
        self,
        seq_length: int = 32,
    ):
        """Construct `IdsMask`.

        This module calculates an attention mask based on the DU, DOM, PMT
        ids between each pair of pulses in the sequence.

        Args:
            seq_length: Dimensionality of the sinusoidal positional embeddings.
        """
        super().__init__()
        self.seq_length = seq_length

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward pass."""
        return (x.unsqueeze(2) == x.unsqueeze(1)).float()
    
class PairwiseProcessing(nn.Module):
    """ Process pairwise features between pulses. """

    def __init__(
                    self,
                    num_masks: int,
                    dims: Union[List, int],
                    num_heads: int,
    ):
        """ Pass all the features through a embedding block before feed them to the model.

            Args:
                in_dims: The number of concatenated pairwise features.
                dims: Dimensionality of the consecutive linear layers.
                num_heads: Number of heads in attention block.
        """

        super().__init__()

        if isinstance(dims, int):
            dims = [dims]

        module_list = []
        for dim in dims:
            module_list.extend([
                                    nn.Conv2d(in_channels = num_masks, out_channels = dim, kernel_size = 1),
                                    nn.BatchNorm2d(dim),
                                    nn.GELU()
            ])
            num_masks = dim
            
        module_list.extend([
                                    nn.Conv2d(in_channels = dims[-1], out_channels = num_heads, kernel_size = 1),
                                    nn.BatchNorm2d(num_heads),
                                    nn.GELU()
        ])
        
        self.emb = nn.Sequential(*module_list)

    def forward(self, x):
        return self.emb(x)

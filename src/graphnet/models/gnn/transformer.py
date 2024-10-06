"""
    Vanilla transformer.
"""
import torch
import math
import torch.nn as nn
from typing import Set, Dict, Any, Optional, Union, List

# Modify here the encoder layers
from graphnet.models.components.layers import (
    Encoder_block,
    NormFormer_block,
)

from graphnet.models.components.embedding import (
    FeaturesProcessing,
    PositionalEncoding,
    PairwiseProcessing,
    CausalityMask,
    EuclideanMask,
    IdsMask,
)

from graphnet.models.gnn.gnn import GNN # Base class for all core GNN models in graphnet.
from graphnet.models.utils import array_to_sequence # Convert `x` of shape [n, d] into a padded sequence of shape [B, L, D].

from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from torch import Tensor


class Transformer(GNN):
    """Vanilla transformer model."""

    def __init__(
        self,
        emb_dims: Union[List, int],
        seq_length: int = 300,
        n_features: int = 8,
        position_encoding: bool = True,
        num_heads: int = 8,
        dropout_attn: float = 0.2,
        hidden_dim: int = 256,
        dropout_FFNN: float = 0.2,
        no_hits_blocks: int = 8,
        no_evt_blocks: Optional[int] = 4,
        ):
        """ Construct a Vanilla Transformer.

        Args:
            seq_length: The total length of the event.
            n_features: The number of features in the input data.
            position_encoder: Wether or not, include position Fourier encoding.
            emb_dims: Embedding dimensions and/or dimension of the model.
            num_heads: Number of heads in MHA.
            dropout_attn: Dropout to be applied in MHA.
            hidden_dim: Dimension of FFNN.
            dropout_FFNN: Dropout to be applied in MHA.
            no_hits_blocks: Number of Encoder blocks using only hit information.
            no_evt_blocks: Number of Encoder blocks including cls token, i.e. considering global event information.
        """
        super().__init__(n_features, emb_dims if isinstance(emb_dims, int) else emb_dims[-1]) #nb_inputs, nb_outputs

        # Take the dimension of the model as the last dimension from emb_dims
        if isinstance(emb_dims, int):
            dim = emb_dims
        elif isinstance(emb_dims, List):
            dim = emb_dims[-1]

        self.n_features = n_features
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Layers:
        self.processing = FeaturesProcessing(emb_dims, n_features)
        self.position_encoding = position_encoding
        self.pos_enc = PositionalEncoding(dim, seq_length)

        self.no_hits_blocks = no_hits_blocks
        self.no_evt_blocks = no_evt_blocks

        self.hits_blocks = nn.Sequential(*[Encoder_block(dim, num_heads, dropout_attn, hidden_dim, dropout_FFNN) for _ in range(no_hits_blocks)])
        self.evt_blocks = nn.Sequential(*[Encoder_block(dim, num_heads, dropout_attn, hidden_dim, dropout_FFNN) for _ in range(no_evt_blocks)])

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_tocken should not be subject to weight decay during training."""
        return {"cls_token"}

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""

        x0, mask0, evt_length = array_to_sequence(
            data.x, data.batch, padding_value=0
        )

        B, L, _ = x0.shape

        x = self.processing(x0)
        cls_token = self.cls_token.repeat(B, 1, 1)

        if self.position_encoding:
            x = self.pos_enc(x)

        mask = torch.zeros(mask0.shape, dtype = mask0.dtype, device = mask0.device)
        mask[~mask0] = -torch.inf

        if self.no_evt_blocks is None or self.no_evt_blocks == 0:
            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.ones((B, 1), dtype = mask0.dtype, device = mask0.device)
            mask = torch.cat([cls_mask, mask], dim=1)

            for hits_block in self.hits_blocks:
                x = hits_block(x, mask=mask)
        else:
            for hits_block in self.hits_blocks:
                x = hits_block(x, mask=mask)

            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.ones((B, 1), dtype = mask0.dtype, device = mask0.device)
            mask = torch.cat([cls_mask, mask], dim=1)

            for evt_block in self.evt_blocks:
                x = evt_block(x, mask=mask)

        return x[:, 0]
    
class PairwiseTransformer(GNN):
    """Vanilla transformer model with pairwise interactions."""

    def __init__(
        self,
        emb_dims: Union[List, int],
        seq_length: int = 300,
        n_features: int = 8,
        position_encoding: bool = True,
        refractive_index: float = 1.33,
        mode: str = 'concat',
        pairwise_dims: Union[List, int] = 64,
        num_heads: int = 8,
        dropout_attn: float = 0.2,
        hidden_dim: int = 256,
        dropout_FFNN: float = 0.2,
        no_hits_blocks: int = 8,
        no_evt_blocks: Optional[int] = 4,
        ):
        """ Construct a Vanilla Transformer with pairwise attention maps.

        Args:
            seq_length: The total length of the event.
            n_features: The number of features in the input data.
            position_encoder: Wether or not, include position Fourier encoding.
            emb_dims: Embedding dimensions and/or dimension of the model.
            num_heads: Number of heads in MHA.
            dropout_attn: Dropout to be applied in MHA.
            hidden_dim: Dimension of FFNN.
            dropout_FFNN: Dropout to be applied in MHA.
            no_hits_blocks: Number of Encoder blocks using only hit information.
            no_evt_blocks: Number of Encoder blocks including cls token, i.e. considering global event information.
        """
        super().__init__(n_features, emb_dims if isinstance(emb_dims, int) else emb_dims[-1]) #nb_inputs, nb_outputs

        # Take the dimension of the model as the last dimension from emb_dims
        if isinstance(emb_dims, int):
            dim = emb_dims
        elif isinstance(emb_dims, List):
            dim = emb_dims[-1]

        self.seq_length = seq_length
        self.n_features = n_features
        self.num_heads = num_heads
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Layers:
        self.processing = FeaturesProcessing(emb_dims, n_features)
        self.position_encoding = position_encoding
        self.pos_enc = PositionalEncoding(dim, seq_length)

        self.no_hits_blocks = no_hits_blocks
        self.no_evt_blocks = no_evt_blocks
        
        self.pw_causality = CausalityMask(seq_length, refractive_index)
        self.pw_euclidean = EuclideanMask(seq_length)
        self.pw_du_ids = IdsMask(seq_length)
        self.pw_dom_ids = IdsMask(seq_length)
        self.pw_pmt_ids = IdsMask(seq_length)
    
        self.mode = mode
        if self.mode == 'concat':
            self.pw_processing = PairwiseProcessing(5, pairwise_dims, num_heads)
        if self.mode == 'sum':
            self.pw_processing = PairwiseProcessing(1, pairwise_dims, num_heads)
    
        self.hits_blocks = nn.Sequential(*[Encoder_block(dim, num_heads, dropout_attn, hidden_dim, dropout_FFNN) for _ in range(no_hits_blocks)])
        self.evt_blocks = nn.Sequential(*[Encoder_block(dim, num_heads, dropout_attn, hidden_dim, dropout_FFNN) for _ in range(no_evt_blocks)])

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_tocken should not be subject to weight decay during training."""
        return {"cls_token"}

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""

        x0, mask0, evt_length = array_to_sequence(
            data.x, data.batch, padding_value=0
        )

        B, L, _ = x0.shape
        
        # Class token creation
        cls_token = self.cls_token.repeat(B, 1, 1)
        
        # Features processing and position encoding [pos, dir, t, tot, du, dom, pmt, trig]
        x = self.processing(x0)
        if self.position_encoding:
            x = self.pos_enc(x)

        # Pairwise features processing 
        x_pos = x0[:, :, 0].unsqueeze(-1)
        y_pos = x0[:, :, 1].unsqueeze(-1)
        z_pos = x0[:, :, 2].unsqueeze(-1)
        t = x0[:, :, 6].unsqueeze(-1)
        du = x0[:, :, 8]
        dom = x0[:, :, 9]
        pmt = x0[:, :, 10]
        
        xt_tensor = torch.cat((x_pos, y_pos, z_pos, t), dim = 2)
        mask_1 = self.pw_causality(xt_tensor).unsqueeze(1)
        x_tensor = torch.cat((x_pos, y_pos, z_pos), dim = 2)
        mask_2 = self.pw_euclidean(x_tensor).unsqueeze(1)
        mask_3 = self.pw_du_ids(du).unsqueeze(1)
        mask_4 = self.pw_dom_ids(dom).unsqueeze(1)
        mask_5 = self.pw_pmt_ids(pmt).unsqueeze(1) 
        
        masks = torch.cat((mask_1, mask_2, mask_3, mask_4, mask_5), dim = 1)
        if self.mode == 'concat':
            attn_mask = masks
        elif self.mode == 'sum':
            attn_mask = torch.sum(masks, dim = 1).unsqueeze(1)
        attn_mask = self.pw_processing(masks) # [BS, num_heads, seq, seq]
        attn_mask = attn_mask.view(B * self.num_heads, L, L) # [BS*num_heads, seq, seq]
            
        # Padding mask
        mask = torch.zeros(mask0.shape, dtype = mask0.dtype, device = mask0.device)
        mask[~mask0] = -torch.inf        

        if self.no_evt_blocks is None or self.no_evt_blocks == 0:
            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.ones((B, 1), dtype = mask0.dtype, device = mask0.device)
            mask = torch.cat([cls_mask, mask], dim=1)

            for hits_block in self.hits_blocks:
                x = hits_block(x, mask=mask, attn_mask = attn_mask)
        else:
            for hits_block in self.hits_blocks:
                x = hits_block(x, mask=mask, attn_mask = attn_mask)

            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.ones((B, 1), dtype = mask0.dtype, device = mask0.device)
            mask = torch.cat([cls_mask, mask], dim=1)

            for evt_block in self.evt_blocks:
                x = evt_block(x, mask=mask)

        return x[:, 0]

    
class NormFormer(GNN):
    """Transformer model inspired by ParticleTransformer: https://arxiv.org/pdf/2202.03772."""

    def __init__(
        self,
        emb_dims: Union[List, int],
        seq_length: int = 300,
        n_features: int = 8,
        position_encoding: bool = True,
        num_heads: int = 8,
        dropout_attn: float = 0.1,
        dropout_attn_cls: float = 0.0,
        hidden_dim: int = 256,
        no_hits_blocks: int = 8,
        no_evt_blocks: Optional[int] = 4,
        ):
        """ Construct a Transformer inspired by ParticleTransformer.

        Args:
            seq_length: The total length of the event.
            n_features: The number of features in the input data.
            position_encoder: Wether or not, include position Fourier encoding.
            emb_dims: Embedding dimensions and/or dimension of the model.
            num_heads: Number of heads in MHA.
            dropout_attn: Dropout to be applied in MHA.
            dropout_attn_cls: Dropout to be applied in MHA when class token is included.
            hidden_dim: Dimension of FFNN.
            no_hits_blocks: Number of NormFormer blocks using only hit information.
            no_evt_blocks: Number of NormFormer blocks including cls token, i.e. considering global event information.
        """
        super().__init__(n_features, emb_dims if isinstance(emb_dims, int) else emb_dims[-1]) #nb_inputs, nb_outputs

        # Take the dimension of the model as the last dimension from emb_dims
        if isinstance(emb_dims, int):
            dim = emb_dims
        elif isinstance(emb_dims, List):
            dim = emb_dims[-1]

        self.n_features = n_features
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Layers:
        self.processing = FeaturesProcessing(emb_dims, n_features)
        self.position_encoding = position_encoding
        self.pos_enc = PositionalEncoding(dim, seq_length)

        self.no_hits_blocks = no_hits_blocks
        self.no_evt_blocks = no_evt_blocks

        self.hits_blocks = nn.Sequential(*[NormFormer_block(dim, num_heads, dropout_attn, hidden_dim) for _ in range(no_hits_blocks)])
        self.evt_blocks = nn.Sequential(*[NormFormer_block(dim, num_heads, dropout_attn_cls, hidden_dim) for _ in range(no_evt_blocks)])

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_tocken should not be subject to weight decay during training."""
        return {"cls_token"}

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""

        x0, mask0, evt_length = array_to_sequence(
            data.x, data.batch, padding_value=0
        )

        B, L, _ = x0.shape

        x = self.processing(x0)
        cls_token = self.cls_token.repeat(B, 1, 1)

        if self.position_encoding:
            x = self.pos_enc(x)

        mask = torch.zeros(mask0.shape, dtype = mask0.dtype, device = mask0.device)
        mask[~mask0] = -torch.inf

        for hits_block in self.hits_blocks:
            x = hits_block(cls_token = None, x = x, mask = mask)
            
        cls_mask = torch.ones((B, 1), dtype = mask0.dtype, device = mask0.device)
        mask = torch.cat([cls_mask, mask], dim=1)

        for evt_block in self.evt_blocks:
            cls_token = evt_block(cls_token = cls_token, x = x, mask = mask)
         
        return cls_token[:, 0]

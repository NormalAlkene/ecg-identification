"""Model
Transformer model
"""

from typing import Callable, override
import math

import torch
from torch import optim, nn, Tensor
import lightning as pl

class PositionalEncoding(nn.Module):
    """Positional encoding
    """
    def __init__(self, dim: int, max_len: int = 5000, base: float = 10000.0, device: str = 'cpu'):
        super().__init__()

        pe: Tensor = torch.zeros(max_len, dim, device = device)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(base) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding
        """
        return x + self.pe[:, :x.size(1)]

class TransformerEcgIdModel(pl.LightningModule):
    """Transformer encoder model for ECG identification.
    """
    device: str
    lr: float

    embedding: nn.Module
    """Linear embedding layer
    In: Tensor[*, feat]
    Out: Tensor[*, feat_]
    """

    positional_encoding: nn.Module
    """Positional encoding
    In: Tensor[batch, seq, feat]
    Out: Tensor[batch, seq, feat]
    """

    encoder: nn.Module
    """Transformer encoder
    In: Tensor[batch, seq, feat]
    Out: Tensor[batch, seq_, feat_]
    """

    fc: nn.Module
    """Linear output layer
    In: Tensor[batch, flattened * 2]
    Out: Tensor[batch, 2]
    """

    criterion: nn.Module
    optimizer: optim.Optimizer

    def __init__(self,
                 num_transformer_layers: int = 4,
                 dim_transformer_layer: int = 512,
                 num_heads: int = 8,
                 dim_input: int = 512,
                 len_seq: int = 10,
                 dropout: float = 0.1,
                 activation: str | Callable[[Tensor], Tensor] = 'relu',
                 num_fc_layers: int = 4,
                 lr: float = 1e-3,
                 device: str = 'cuda',
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.device = device
        self.len_seq = len_seq
        self.lr = lr

        self.embedding = nn.Linear(dim_input, dim_transformer_layer)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = dim_transformer_layer,
                nhead = num_heads,
                dropout = dropout,
                activation = activation,
                batch_first = True,
                device = device
                ),
            num_layers = num_transformer_layers
        )

        fc_list: list[nn.Module] = []
        for _ in range(num_fc_layers):
            fc_list.append(nn.Linear(dim_transformer_layer * len_seq, dim_transformer_layer * len_seq))
            fc_list.append(nn.LeakyReLU())

        fc_list.append(nn.Linear(dim_transformer_layer * len_seq, 2))
        fc_list.append(nn.Sigmoid())
        self.fc = nn.Sequential(*fc_list)

        self.criterion = nn.BCELoss()

    @override
    def forward(self, batch: list[Tensor]) -> Tensor:
        """Forward

        Args:
            batch (list[Tensor]): batch data
                batch[0] (Tensor[batch, seq, feat]): input 0
                batch[1] (Tensor[batch, seq, feat]): input 1 
                batch[2] (Tensor[batch, feat = 2]): label

            batch_idx (Any): _description_

        Returns:
            Any: _description_
        """

        inputs: list[Tensor] = [batch[0], batch[1]]
        inputs = map(
            lambda x:
                self.encoder(
                    self.positional_encoding(
                        self.embedding(x)
                    )
                ).flatten(start_dim = 1),
            inputs
        )

        # memory: Tensor[batch, i, seq, feat]
        memory: Tensor = torch.cat(inputs, dim = 1)

        return self.fc(memory)

    @override
    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        """Training step
        """
        outputs: Tensor = self(batch)
        loss: Tensor = self.criterion(outputs, batch[2])

        self.log('train_loss', loss)
        return loss

    @override
    def validation_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        """Validation step
        """
        outputs: Tensor = self(batch)
        loss: Tensor = self.criterion(outputs, batch[2])

        self.log('val_loss', loss)
        return loss

    @override
    def testing_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        """Testing step
        """
        outputs: Tensor = self(batch)
        loss: Tensor = self.criterion(outputs, batch[2])

        self.log('test_loss', loss)
        return loss

    @override
    def configure_optimizers(self) -> optim.Optimizer:
        """Configure optimizer
        """
        return optim.Adam(self.parameters(), lr = self.lr)

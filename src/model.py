"""Model
Transformer model
"""

from typing import Callable, override
import math

import torch
from torch import optim, nn, Tensor
from torchmetrics.classification import BinaryAccuracy
import lightning as pl

class PositionalEncoding(nn.Module):
    """Positional encoding
    """
    def __init__(self, dim: int, max_len: int = 5000, base: float = 10000.0, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()

        pe: Tensor = torch.zeros(max_len, dim, device = device, dtype = dtype)
        position = torch.arange(0, max_len, dtype = dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, step = 2, dtype = dtype) * -(math.log(base) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding
        """
        return x + self.pe[:, :x.size(1)]

class TransformerEcgIdModel(pl.LightningModule):
    """Transformer encoder model for ECG identification.
    """
    _lr: float
    _lr_gamma: float
    _len_seq: int

    _embedding: nn.Module
    """Linear embedding layer
    In: Tensor[*, feat]
    Out: Tensor[*, feat_]
    """

    _pe: nn.Module
    """Positional encoding
    In: Tensor[batch, seq, feat]
    Out: Tensor[batch, seq, feat]
    """

    _encoder: nn.Module
    """Transformer encoder
    In: Tensor[batch, seq, feat]
    Out: Tensor[batch, seq_, feat_]
    """

    _fc: nn.Module
    """Linear output layer
    In: Tensor[batch, flattened * 2]
    Out: Tensor[batch, 2]
    """

    _criterion: nn.Module
    _metric: nn.Module

    def __init__(self,
                 num_transformer_layers: int = 4,
                 dim_transformer_layer: int = 512,
                 num_heads: int = 8,
                 dim_input: int = 512,
                 len_seq: int = 10,
                 dropout: float = 0.1,
                 activation: str | Callable[[Tensor], Tensor] = "relu",
                 num_fc_layers: int = 4,
                 lr: float = 1e-3,
                 lr_gamma: float = 0.95,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self._len_seq = len_seq
        self._lr = lr
        self._lr_gamma = lr_gamma

        self._embedding = nn.Linear(dim_input, dim_transformer_layer, dtype = dtype)
        self._pe = PositionalEncoding(dim_transformer_layer, len_seq, device = device, dtype = dtype)
        self._encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = dim_transformer_layer,
                nhead = num_heads,
                dropout = dropout,
                activation = activation,
                batch_first = True,
                device = device,
                dtype = dtype
            ),
            num_layers = num_transformer_layers
        )

        fc_list: list[nn.Module] = []
        for _ in range(num_fc_layers):
            fc_list.append(nn.Linear(dim_transformer_layer * len_seq * 2, dim_transformer_layer * len_seq * 2, dtype = dtype))
            fc_list.append(nn.LeakyReLU(inplace = True))

        fc_list.append(nn.Linear(dim_transformer_layer * len_seq * 2, 2, dtype = dtype))
        fc_list.append(nn.Sigmoid())
        self._fc = nn.Sequential(*fc_list)

        self._criterion = nn.BCELoss()
        self._metric = BinaryAccuracy(multidim_average = "samplewise")

    @override
    def forward(self, batch: list[Tensor]) -> Tensor:
        """Forward

        Args:
            batch (list[Tensor]): batch data
                batch[0] (Tensor[batch, seq, feat]): input 0
                batch[1] (Tensor[batch, seq, feat]): input 1 
                batch[2] (Tensor[batch, feat = 2]): label

            batch_idx (Any):

        Returns:
            Tensor: the result
        """

        encoded = [
            torch.flatten(self._encoder(self._pe(self._embedding(batch[0]))), start_dim = 1),
            torch.flatten(self._encoder(self._pe(self._embedding(batch[1]))), start_dim = 1)
        ] # type: ignore

        # Tensor[batch, flattened(seq, feat) * 2]
        encoded = torch.cat(encoded, dim = 1)

        return self._fc(encoded)

    @override
    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        """Training step
        """
        outputs: Tensor = self(batch)
        loss: Tensor = self._criterion(outputs, batch[2])

        self.log("train_loss", loss.item(), prog_bar = True)
        return loss

    @override
    def validation_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        """Validation step
        """
        outputs: Tensor = self(batch)
        loss: Tensor = self._criterion(outputs, batch[2])
        acc: Tensor = self._metric(outputs.transpose(0, 1), batch[2].transpose(0, 1))

        self.log("val_loss", loss.item(), prog_bar = True)
        self.log("acc_0", acc[0].item(), prog_bar = True)
        self.log("acc_1", acc[1].item(), prog_bar = True)

        return loss

    @override
    def test_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        """Testing step
        """
        outputs: Tensor = self(batch)
        loss: Tensor = self._criterion(outputs, batch[2])

        self.log("test_loss", loss.item(), prog_bar = True)
        return loss

    @override
    def configure_optimizers(self) -> optim.Optimizer:
        """Configure optimizer
        """
        optimizer = optim.RAdam(self.parameters(), lr = self._lr)
        scheduler = {
            "scheduler": optim.lr_scheduler.ExponentialLR(optimizer, self._lr_gamma),
            "interval": "epoch"
        }
        return [optimizer], [scheduler]


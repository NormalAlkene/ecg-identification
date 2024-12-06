#!/usr/bin/env python3
"""
The config file of ECG identification
"""

from dataclasses import dataclass
import os

import torch

DEVICE: str = 'cuda'

@dataclass
class _Paths:
    path_ds_raw: str = 'data/0-raw'
    path_ds_preprocessed: str = 'data/1-preprocessed'
    path_ds_training: str = 'data/2-training'
    path_ds_validation: str = 'data/2-validation'
    path_ds_testing: str = 'data/2-testing'
    path_results: str = 'data/3-results'
    path_model: str = 'model'
    name_ds_training: str = 'training.npz'
    name_ds_validation: str = 'validation.npz'
    name_ds_testing: str = 'testing.npz'

PATHS = _Paths()

@dataclass
class _Preprocessing:
    fs: float = 512.0
    detrend_order: int = 9
    lowcut: float = 0.5
    highcut: float = 256 # of no use
    ratio_validation: float = 0.15
    ratio_testing: float = 0.15

PREPROCESSING = _Preprocessing()

@dataclass
class _TransformerModel:
    num_transformer_layers: int = 4
    dim_transformer_layer: int = 512
    num_heads: int = 8
    dim_input: int = 512
    len_seq: int = 100
    dropout: float = 0.1
    activation: str = 'relu'
    num_fc_layers: int = 1
    lr: float = 1e-3
    device: str = 'cuda'
    dtype: torch.dtype  = torch.float32

MODEL_TRANSFORMER = _TransformerModel()

@dataclass
class _Trainer:
    max_epochs: int = 128
    accelerator: str = 'cuda'
    log_every_n_steps: int = 10


TRAINER = _Trainer()

@dataclass
class _Dataset:
    sample_len: int = MODEL_TRANSFORMER.len_seq * MODEL_TRANSFORMER.dim_input
    token_len: int = MODEL_TRANSFORMER.dim_input
    device: str = DEVICE
    dtype: torch.dtype = MODEL_TRANSFORMER.dtype

DATASET = _Dataset()

@dataclass
class _Dataloader:
    batch_size: int = 1
    num_workers: int = 0
    shuffle: bool = False
    drop_last: bool = False

DATALOADER_TRAINING = _Dataloader(batch_size=32, num_workers=0, shuffle=True, drop_last=True)
DATALOADER_VALIDATION = _Dataloader(batch_size=32, num_workers=0, shuffle=True, drop_last=True)
DATALOADER_TESTING = _Dataloader(batch_size=1, num_workers=0)
#DATALOADER_TRAINING = _Dataloader(batch_size=32, num_workers=os.cpu_count() - 1, shuffle=True)
#DATALOADER_VALIDATION = _Dataloader(batch_size=1, num_workers=os.cpu_count() - 1)
#DATALOADER_TESTING = _Dataloader(batch_size=1, num_workers=os.cpu_count() - 1)

@dataclass
class _Checkpoint:
    dirpath: str = PATHS.path_model
    every_n_epochs: int = 1
    save_on_train_epoch_end: bool = True
    save_top_k: int = -1
    verbose: bool = True

CHECKPOINT = _Checkpoint()

@dataclass
class _EarlyStopping:
    monitor: str = 'val_loss'
    patience: int = 3
    mode: str = 'min'

EARLY_STOPPING = _EarlyStopping()

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
    num_transformer_layers: int = 6
    dim_transformer_layer: int = 8
    num_heads: int = 4
    dim_input: int = 16
    len_seq: int = 128
    dropout: float = 0.05
    activation: str = 'relu'
    num_fc_layers: int = 1
    lr: float = 1e-3
    lr_gamma: float = 0.9
    device: str = DEVICE
    dtype: torch.dtype  = torch.float32

MODEL_TRANSFORMER = _TransformerModel()

@dataclass
class _Trainer:
    max_epochs: int = 1024
    accelerator: str = DEVICE
    log_every_n_steps: int = 20

TRAINER = _Trainer()

@dataclass
class _Dataset:
    sample_len: int = 0
    token_len: int = 0
    dtype: torch.dtype = 0

DATASET_TRANSFORMER = _Dataset(
    sample_len = MODEL_TRANSFORMER.len_seq * MODEL_TRANSFORMER.dim_input,
    token_len = MODEL_TRANSFORMER.dim_input,
    dtype = MODEL_TRANSFORMER.dtype
)

@dataclass
class _Dataloader:
    batch_size: int = 1
    num_workers: int = 0
    shuffle: bool = False
    drop_last: bool = False

#DATALOADER_TRAINING = _Dataloader(batch_size=512, num_workers=0, shuffle=True, drop_last=False)
#DATALOADER_VALIDATION = _Dataloader(batch_size=512, num_workers=0, shuffle=True, drop_last=False)
#DATALOADER_TESTING = _Dataloader(batch_size=1, num_workers=0)
DATALOADER_TRAINING = _Dataloader(batch_size = 150, num_workers = os.cpu_count() - 1, shuffle = True, drop_last = True)
DATALOADER_VALIDATION = _Dataloader(batch_size = 150, num_workers = os.cpu_count() - 1, drop_last = False)
DATALOADER_TESTING = _Dataloader(batch_size = 512, num_workers = os.cpu_count() - 1, drop_last = False)

@dataclass
class _Checkpoint:
    dirpath: str = PATHS.path_model
    every_n_epochs: int = 2
    save_on_train_epoch_end: bool = True
    save_top_k: int = -1
    verbose: bool = True

CHECKPOINT = _Checkpoint()

@dataclass
class _EarlyStopping:
    monitor: str = 'val_loss'
    patience: int = 8
    mode: str = 'min'

EARLY_STOPPING = _EarlyStopping()

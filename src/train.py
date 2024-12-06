#!/usr/bin/env python3

"""Train the model"""

import os
from dataclasses import asdict
import multiprocessing

from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from model import TransformerEcgIdModel
from dataset import EcgIdDataset, ResampleCallback
from configs import *

def main() -> None:
    """main"""
    #multiprocessing.set_start_method("forkserver")
    torch.set_float32_matmul_precision("medium")
    model = TransformerEcgIdModel(**asdict(MODEL_TRANSFORMER))
    #model = torch.compile(model, fullgraph = False) # will cause exceptions
    ds_training = EcgIdDataset(
        os.path.join(PATHS.path_ds_training, PATHS.name_ds_training),
        **asdict(DATASET)
    )
    ds_validation = EcgIdDataset(
        os.path.join(PATHS.path_ds_validation, PATHS.name_ds_validation),
        **asdict(DATASET)
    )
    dl_training = DataLoader(ds_training, **asdict(DATALOADER_TRAINING))
    dl_validation = DataLoader(ds_validation, **asdict(DATALOADER_VALIDATION))

    logger = TensorBoardLogger(PATHS.path_model)

    trainer = pl.Trainer(
        logger = logger,
        callbacks = [
            ResampleCallback(),
            ModelCheckpoint(**asdict(CHECKPOINT)),
            EarlyStopping(**asdict(EARLY_STOPPING))
        ],
        **asdict(TRAINER)
    )
    trainer.fit(model, dl_training, dl_validation)

if __name__ == "__main__":
    main()

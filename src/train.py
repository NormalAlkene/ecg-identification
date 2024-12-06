#!/usr/bin/env python3

"""Train the model"""

import os
from dataclasses import asdict

from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from model import TransformerEcgIdModel
from dataset import EcgIdDataset, ResampleCallback
from configs import *

def main() -> None:
    """main"""
    model = TransformerEcgIdModel(**asdict(MODEL_TRANSFORMER))
    ds_training = EcgIdDataset(
        os.path.join(PATHS.path_ds_training, PATHS.name_ds_training),
        **asdict(DATASET)
    )
    ds_validation = EcgIdDataset(
        os.path.join(PATHS.path_ds_validation, PATHS.name_ds_validation),
        **asdict(DATASET)
    )
    dl_training = DataLoader(ds_training, DATALOADER_TRAINING)
    dl_validation = DataLoader(ds_validation, DATALOADER_VALIDATION)

    logger = TensorBoardLogger(PATHS.path_model)

    trainer = pl.Trainer(
        logger = logger,
        callbacks = [
            ResampleCallback(),
            ModelCheckpoint(**asdict(CHECKPOINT)),
            EarlyStopping(**asdict(EARLY_STOPPING))
        ]
        **asdict(TRAINER)
    )
    trainer.fit(model, dl_training, dl_validation)

if __name__ == '__main__':
    main()

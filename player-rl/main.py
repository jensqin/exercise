import os
from argparse import ArgumentParser
from datetime import datetime
import math

import numpy as np
import pandas as pd
import awswrangler as wr
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

from utils import S3_INPUT_FOLDER, load_nba, train_val_test_split
from modules.core import NBADataModule, NBAEncoder, save_output
from models.ridge import NBARidge
from models.mlr import NBAMixedLogit, NBARidgeMLR, NBAMLRShootEmb
from models.dcn import NBADCN
from models.former import NBATransformer, NBAShootTF


def load_nba_data(test_size=0.2):
    df = wr.s3.read_parquet(S3_INPUT_FOLDER + "zb_possession_example")
    df[
        [
            "OffTeam",
            "DefTeam",
            "P1",
            "P2",
            "P3",
            "P4",
            "P5",
            "P6",
            "P7",
            "P8",
            "P9",
            "P10",
        ]
    ] = df[
        [
            "OffTeam",
            "DefTeam",
            "P1",
            "P2",
            "P3",
            "P4",
            "P5",
            "P6",
            "P7",
            "P8",
            "P9",
            "P10",
        ]
    ].astype(
        "int64"
    )
    # df = df[df["Season"] >= 2020]
    split_nrow = math.floor(len(df.index) * (1 - test_size))
    train = df[0:split_nrow]
    test = df[split_nrow:-1]
    return train, test


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out_file", type=str, default="")
    parser = NBADataModule.add_data_specific_args(parser)
    parser = NBAEncoder.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    nba_early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min"
    )
    dict_args = vars(args)
    pl.seed_everything(0)
    train, test = load_nba_data()
    nba = NBADataModule(train=train, test=test, loss="mse", **dict_args)
    model = NBARidge(loss="mse", **dict_args)
    tb_logger = TensorBoardLogger(save_dir=args.logdir)
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        callbacks=[nba_early_stopping],
        max_epochs=args.epochs,
        # precision=16,
    )
    start = datetime.now()
    trainer.fit(model, datamodule=nba)
    trainer.save_checkpoint("ckpt/nba_2018.ckpt")
    save_output(model, args.out_file)
    print(datetime.now() - start)
    # os.system('say "Your Python Program has Finished"')

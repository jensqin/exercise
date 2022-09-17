import os
from typing import Tuple
from dotenv import find_dotenv, load_dotenv
from bla_python_db_utilities import auth

import numpy as np
import pandas as pd
import torch
import awswrangler as wr
from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split

root_dir = "~/repository/exercise/ridge_reg"

S3_INPUT_FOLDER = "s3://bla-basketball-models/processing/"
S3_OUTPUT_FOLDER = "s3://bla-basketball-models/processing/"


def train_val_test_split(
    df, val=0.15, test=0.1, shuffle=True, stratify_cols=None, random_state=None
):
    """
    training validation test data split
    """
    df_train, df_test = train_test_split(
        df,
        test_size=test,
        shuffle=shuffle,
        stratify=df[stratify_cols],
        random_state=random_state,
    )
    df_train, df_val = train_test_split(
        df_train,
        test_size=val,
        shuffle=shuffle,
        stratify=df_train[stratify_cols],
        random_state=random_state,
    )
    return df_train, df_val, df_test


def load_nba(
    path=S3_INPUT_FOLDER + "zb_possession",
    split_mode=None,
    test=0.1,
    val=0.15,
    to_tensor=False,
):
    """
    load nba data
    """
    float_cols = ["Pts", "HomeOff", "ScoreMargin"] + [
        "Age1",
        "Age2",
        "Age3",
        "Age4",
        "Age5",
        "Age6",
        "Age7",
        "Age8",
        "Age9",
        "Age10",
    ]
    type_dict = {key: np.float32 for key in float_cols}
    # path = os.path.join(root_dir, path)
    df = pd.read_csv(path, dtype=type_dict, index_col=False)
    stratify_cols = ["OffTeam", "DefTeam"]
    if split_mode is None:
        if to_tensor:
            return transform_to_array(df)
        else:
            return df
    elif split_mode == "test":
        train, test = train_test_split(df, test_size=test, stratify=df[stratify_cols])
        if to_tensor:
            train, test = transform_to_array(train), transform_to_array(test)
        return train, test
    else:
        train, val, test = train_val_test_split(
            df, test=test, val=val, stratify_cols=stratify_cols, shuffle=True
        )
        if to_tensor:
            train, val, test = (
                transform_to_array(train),
                transform_to_array(val),
                transform_to_array(test),
            )
        return train, val, test


def load_nba_sparse(data="train"):
    """
    load nba sparse data
    """
    x_path = f"data/nba_2018_sparse/X_{data}.csv"
    y_path = f"data/nba_2018_sparse/y_{data}.csv"
    dfx = pd.read_csv(x_path)
    x = csc_matrix((dfx["x"], (dfx["i"] - 1, dfx["j"] - 1)))
    dfy = pd.read_csv(y_path, dtype={"y": np.float64})
    return x, dfy["y"].values


def transform_to_array(df, to_tensor=True):
    """
    transform nba data to tensors
    """
    df = df.reset_index(drop=True)
    x_cols = [
        ["HomeOff", "ScoreMargin"],
        ["OffTeam"],
        ["DefTeam"],
        ["P1", "P2", "P3", "P4", "P5"],
        ["Age1", "Age2", "Age3", "Age4", "Age5"],
        ["P6", "P7", "P8", "P9", "P10"],
        ["Age6", "Age7", "Age8", "Age9", "Age10"],
    ]
    y_cols = ["Pts"]
    x = [df[col].values for col in x_cols]
    y = df[y_cols].values
    if to_tensor:
        x = [torch.from_numpy(t) for t in x]
        y = torch.from_numpy(y)
    return x, y


def summary_samples(samples):
    return {
        k: {
            "mean": torch.mean(v, 0).detach().numpy(),
            "std": torch.std(v, 0).detach().numpy(),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0].detach().numpy(),
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0].detach().numpy(),
        }
        for k, v in samples.items()
    }


load_dotenv(find_dotenv(), override=True)

ENGINE_CONFIG = auth.load_url_dict(
    prefix=[
        "BASKETBALL_NBA_DEV.",
        "BASKETBALL_NBA_MODEL_DEV.",
        "BASKETBALL_NBA_TEST.",
        "BASKETBALL_NBA_MODEL_TEST.",
    ]
)

SQL_DIR = "sql"
SQL_PATH = {"sportradar": "sportradar.sql"}
SQL_PATH = {key: SQL_DIR + "/" + value for key, value in SQL_PATH.items()}

S3_BUCKET = "bla-basketball-models"
S3_FOLDER = "s3://bla-basketball-models/processing/"
HPARAMS_URL = "s3://" + S3_BUCKET + "/NBA/zqin-models/hyperparameters"
HPARAMS_PATH = "nba/hparams.json"
MODEL_URL = "NBA/zqin-models/models"
DATALOADER_URL = "NBA/zqin-models/dataloaders"

LOG_DIR = "nba/.tensorboard_logs"


def download_s3_data():
    """download data from s3"""
    df = wr.s3.read_parquet("s3://bla-basketball-models/processing/zb_play")
    df = df.astype(
        {
            x: "int"
            for x in [
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
                "OffTeam",
                "DefTeam",
                "UsageId",
            ]
        }
    )
    return df

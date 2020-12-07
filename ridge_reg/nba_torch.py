from argparse import ArgumentParser
from datetime import datetime
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW, Adamax
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils import train_val_test_split


class NBADataset(Dataset):
    """
    NBA NW Data
    """

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        # print(self.df.dtypes)
        self.idx = df.index
        self.x = [
            ["HomeAway", "ScoreDiff"],
            ["OffTeam"],
            ["DefTeam"],
            ["P1", "P2", "P3", "P4", "P5"],
            ["age1", "age2", "age3", "age4", "age5"],
            ["P6", "P7", "P8", "P9", "P10"],
            ["age6", "age7", "age8", "age9", "age10"],
        ]
        self.y = ["y"]

    def __len__(self):
        return len(self.idx)

    @staticmethod
    def pd_to_tensor(x):
        xarray = x.values[0]
        return torch.from_numpy(xarray)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif isinstance(idx, int):
            idx = [idx]
        sample_df = self.df.iloc[idx, :]
        return (
            [self.pd_to_tensor(sample_df[col]) for col in self.x],
            self.pd_to_tensor(sample_df[self.y]),
        )


class NBADataModule(pl.LightningDataModule):
    """
    NBA NW Division data
    """

    def __init__(
        self, data_path="data/nba_nw.csv", num_workders=1, batch_size=32, **kwargs
    ):
        super().__init__()
        self.data_path = os.path.join("~/repository/exercise/ridge_reg", data_path)
        self.num_workders = num_workders
        self.batch_size = batch_size
        self.stratify_cols = ["OffTeam", "DefTeam"]
        self.float_cols = ["y", "HomeAway", "ScoreDiff"] + [
            f"age{x}" for x in range(1, 11)
        ]

    def setup(self, stage=None):
        type_dict = {key: np.float32 for key in self.float_cols}
        # type_dict.update({"y": np.int32})
        self.nba = pd.read_csv(self.data_path, dtype=type_dict)
        train, val, test = train_val_test_split(
            self.nba, shuffle=True, stratify_cols=self.stratify_cols, random_state=None
        )
        self.train = NBADataset(train)
        self.val = NBADataset(val)
        self.test = NBADataset(test)

    def train_dataloader(self):
        """
        training dataloader
        """
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workders,
            drop_last=True,
        )

    def val_dataloader(self):
        """
        validation dataloader
        """
        return self.build_dataloader(self.val)

    def test_dataloader(self):
        """
        test dataloader
        """
        return self.build_dataloader(self.test)

    def build_dataloader(self, dataset):
        """
        build dataloader
        """
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workders
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        """
        model specific args
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_workders", type=int, default=4)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--data_path", type=str, default="data/nba_nw.csv")
        return parser


class NBAEncoder(pl.LightningModule):
    """
    NBA Encoders
    """

    def __init__(
        self,
        # team_data_path="data/team_map.csv",
        # player_data_path="data/player_map.csv",
        lr=0.1,
        weight_decay=[0.0],
        **kwargs,
    ):
        super().__init__()
        # n_team, n_player = NBAEncoder.n_team_and_player(
        #     team_data_path, player_data_path
        # )
        n_team, n_player = 5, 85
        n_team_emb, n_player_emb = 1, 1
        self.lr = lr
        self.wd = weight_decay
        self.fc = nn.Linear(2, 1)
        self.off_team = nn.Embedding(n_team, n_team_emb)
        self.def_team = nn.Embedding(n_team, n_team_emb)
        # self.dense = nn.ModuleList([self.fc, self.off_team, self.def_team])
        self.off_player = nn.EmbeddingBag(n_player, n_player_emb, mode="sum")
        self.off_player_age = nn.EmbeddingBag(n_player, n_player_emb, mode="sum")
        self.def_player = nn.EmbeddingBag(n_player, n_player_emb, mode="sum")
        self.def_player_age = nn.EmbeddingBag(n_player, n_player_emb, mode="sum")
        # player_dim = n_player_emb * 4
        # self.bn_player = nn.BatchNorm1d(player_dim)
        # self.fin = nn.Linear(2 + n_team_emb * 2 + n_player_emb * 4, 3)
        # self.sparse = nn.ModuleList(
        #     [self.off_player, self.off_player_age, self.def_player, self.def_player_age]
        # )

    @staticmethod
    def n_team_and_player(team_data_path, player_data_path):
        """
        number of teams and number of players
        """
        team = pd.read_csv(team_data_path)
        player = pd.read_csv(player_data_path)
        n_team = team["teamids"].nunique()
        n_player = player["playerids"].nunique()
        return n_team, n_player


class NBAGroupRidge(NBAEncoder):
    """
    NBA Representation Learning
    """

    # def __init__(
    #     self,
    #     team_data_path="data/team_map.csv",
    #     player_data_path="data/player_map.csv",
    #     lr=0.1,
    #     weight_decay=[0.0],
    #     **kwargs,
    # ):
    #     super().__init__()
    #     n_team, n_player = NBAEncoder.n_team_and_player(team_data_path, player_data_path)
    #     self.lr = lr
    #     self.wd = weight_decay
    #     self.fc = nn.Linear(2, 1, bias=True)
    #     self.off_team = nn.Embedding(n_team, 1)
    #     self.def_team = nn.Embedding(n_player, 1)
    #     # self.dense = nn.ModuleList([self.fc, self.off_team, self.def_team])
    #     self.off_player = nn.EmbeddingBag(n_player, 1, mode="sum")
    #     self.off_player_age = nn.EmbeddingBag(n_player, 1, mode="sum")
    #     self.def_player = nn.EmbeddingBag(n_player, 1, mode="sum")
    #     self.def_player_age = nn.EmbeddingBag(n_player, 1, mode="sum")
    #     # self.sparse = nn.ModuleList(
    #     #     [self.off_player, self.off_player_age, self.def_player, self.def_player_age]
    #     # )

    def forward(self, x, return_embedding=True):
        """
        representations
        """
        fc, offt, deft, offp, offpa, defp, defpa = x
        fc = self.fc(fc)
        offt = self.off_team(offt).view(-1, 1)
        deft = self.def_team(deft).view(-1, 1)
        offpa = self.off_player_age(offp, per_sample_weights=offpa)
        defpa = self.def_player_age(defp, per_sample_weights=defpa)
        offp = self.off_player(offp)
        defp = self.def_player(defp)

        if return_embedding:
            return torch.cat([offt, deft, offp, offpa, defp, defpa], dim=1)
        return fc + offt + deft + offpa + defpa + offp + defp

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        training step
        """
        loss = self.mse_loss(batch)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        """
        training epoch end
        """
        logger = self.logger.experiment
        logger.add_histogram("fc_weight", self.fc.weight, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        """
        validation step
        """
        loss = self.mse_loss(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        """
        test step
        """
        return self.mse_loss(batch)

    def configure_optimizers(self):
        """
        configure optimizer
        """
        # opt_param = [
        #     {"params": self.fc.parameters(), "weight_decay": 0},
        #     {"params": self.off_team.parameters(), "weight_decay": self.wd[0]},
        #     {"params": self.def_team.parameters(), "weight_decay": self.wd[1]},
        #     {"params": self.off_player.parameters(), "weight_decay": self.wd[2]},
        #     {"params": self.off_player_age.parameters(), "weight_decay": self.wd[3],},
        #     {"params": self.def_player.parameters(), "weight_decay": self.wd[4]},
        #     {"params": self.def_player_age.parameters(), "weight_decay": self.wd[5]},
        # ]
        dense_param = [
            {"params": self.fc.parameters(), "weight_decay": 0},
            {"params": self.off_team.parameters(), "weight_decay": self.wd[0]},
            {"params": self.def_team.parameters(), "weight_decay": self.wd[1]},
        ]
        sparse_param = [
            {"params": self.off_player.parameters(), "weight_decay": self.wd[2]},
            {"params": self.off_player_age.parameters(), "weight_decay": self.wd[3]},
            {"params": self.def_player.parameters(), "weight_decay": self.wd[4]},
            {"params": self.def_player_age.parameters(), "weight_decay": self.wd[5]},
        ]
        # optimizer = SGD(opt_param, lr=self.lr, momentum=0.8)
        dense_opt = AdamW(dense_param, lr=self.lr)
        sparse_opt = Adamax(sparse_param, lr=self.lr)
        dense_scheduler = OneCycleLR(dense_opt, max_lr=0.1, total_steps=500)
        sparse_scheduler = OneCycleLR(sparse_opt, max_lr=0.1, total_steps=500)
        return [dense_opt, sparse_opt], [dense_scheduler, sparse_scheduler]

    def mse_loss(self, batch):
        """
        calculate mse loss
        """
        x, y = batch
        yhat = self(x, return_embedding=False)
        return F.mse_loss(torch.flatten(yhat), torch.flatten(y))

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        model specific args
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=0.05)
        parser.add_argument("--team_data_path", type=str, default="data/team_map.csv")
        parser.add_argument(
            "--player_data_path", type=str, default="data/player_map.csv"
        )
        parser.add_argument("--weight_decay", type=float, nargs="+", default=[0.05] * 6)
        return parser

    @staticmethod
    def n_team_and_player(team_data_path, player_data_path):
        """
        number of teams and number of players
        """
        team = pd.read_csv(team_data_path)
        player = pd.read_csv(player_data_path)
        n_team = team["teamids"].nunique()
        n_player = player["playerids"].nunique()
        return n_team, n_player


# tmp = None
# tmp1 = DataLoader(NBADataset(tmp), batch_size=3)
# tmp2, tmp3 = next(iter(tmp1))
# tmp2[3]
# tmpp = NBAEncoderModule()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logs")
    parser = NBADataModule.add_data_specific_args(parser)
    parser = NBAGroupRidge.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    nba_early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min"
    )
    dict_args = vars(args)
    nba = NBADataModule(**dict_args)
    model = NBAGroupRidge(**dict_args)
    tb_logger = TensorBoardLogger(save_dir=args.logdir)
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        callbacks=[nba_early_stopping],
        max_epochs=5,
        # precision=16,
    )
    start = datetime.now()
    trainer.fit(model, datamodule=nba)
    print(datetime.now() - start)

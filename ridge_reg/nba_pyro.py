from argparse import ArgumentParser

import pyro
import torch
from torch import nn
from pyro.nn import PyroModule
import pytorch_lightning as pl
import pandas as pd
from torch.nn import functional as F
from torch.optim import AdamW, Adamax
from torch.optim.lr_scheduler import OneCycleLR
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from nba_torch import NBAEncoder, NBADataModule

class NBABayesEncoder(pl.LightningModule):
    """
    NBA hierachical bayesian encoder
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
        self.off_team = PyroModule[nn.Embedding](n_team, n_team_emb)
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
        parser.add_argument("--weight_decay", type=float, nargs="+", default=[0.0] * 6)
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
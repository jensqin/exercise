from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import functional as F
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adamax, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from modules.core import NBADataModule, NBAEncoder, ResBlock
from utils import load_nba, train_val_test_split


class NBAResNet(NBAEncoder):
    """
    mixed logistic regression
    """

    def __init__(
        self,
        # team_data_path="data/team_map.csv",
        # player_data_path="data/player_map.csv",
        lr=0.1,
        weight_decay=[0.0],
        **kwargs,
    ):
        """
        init method
        """
        super().__init__(lr, weight_decay)
        offdef_dim = self.n_player_emb * 2
        tp_dim = self.n_team_emb + offdef_dim
        self.off = nn.Linear(self.n_player_emb * 2, self.n_player_emb * 2)
        self.deff = nn.Linear(self.n_player_emb * 2, self.n_player_emb * 2)
        self.offtp = nn.Linear(tp_dim, tp_dim)
        self.deftp = nn.Linear(tp_dim, tp_dim)
        # self.mix = nn.Bilinear(tp_dim, tp_dim, tp_dim * 2)
        self.resblock = ResBlock(self.in_dim, self.in_dim)
        self.threshold = nn.Threshold(0, 0)
        # self.fc1 = nn.Linear(in_dim, in_dim)
        # player_dim = n_player_emb * 4
        # self.bn_player = nn.BatchNorm1d(player_dim)
        # self.fin = nn.Linear(2 + n_team_emb * 2 + n_player_emb * 4, 3)
        # self.sparse = nn.ModuleList(
        #     [self.off_player, self.off_player_age, self.def_player, self.def_player_age]
        # )

    def forward(self, x, return_embedding=True):
        """
        representations
        """
        fc, offt, deft, offp, offpa, defp, defpa = x
        # fc = self.fc(fc)
        offt = self.off_team(offt).view(-1, self.n_team_emb)
        deft = self.def_team(deft).view(-1, self.n_team_emb)
        offpa = self.off_player_age(offp, per_sample_weights=offpa)
        defpa = self.def_player_age(defp, per_sample_weights=defpa)
        offp = self.off_player(offp)
        defp = self.def_player(defp)
        off = torch.cat([offp, offpa], dim=1)
        off = self.off(off)
        off = F.silu(off)
        offtp = torch.cat([offt, off], dim=1)
        offtp = self.offtp(offtp)
        offtp = F.celu(offtp)
        deff = torch.cat([defp, defpa], dim=1)
        deff = self.deff(deff)
        deff = F.silu(deff)
        deftp = torch.cat([deft, deff], dim=1)
        deftp = self.deftp(deftp)
        deftp = F.celu(deftp)
        # mix = self.mix(offtp, deftp)

        # offdef = self.mix(off, deff)
        # offdef = F.selu(offdef)
        # emb = torch.cat([fc, offt, deft, offp, offpa, defp, defpa], dim=1)
        # emb = torch.cat([fc, offt, deft, off, deff], dim=1)
        emb = torch.cat([fc, offtp, deftp], dim=1)
        emb = self.resblock(emb)
        emb = F.celu(emb)

        if return_embedding:
            return torch.cat([offt, deft, offp, offpa, defp, defpa], dim=1)
        # return fc + offt + deft + offpa + defpa + offp + defp
        region_dist = self.region(emb)
        made_p = torch.sigmoid(self.made_p(emb))
        result = torch.sum(region_dist * made_p * torch.linspace(0, 4, 9), dim=-1)
        # made_p = self.made_p(emb)
        # result = region_dist * made_p
        # result = torch.sum(
        #     region_dist * made_p, dim=-1
        # )
        return self.threshold(result)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--epochs", type=int, default=5)
    parser = NBADataModule.add_data_specific_args(parser)
    parser = NBAEncoder.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    nba_early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min"
    )
    dict_args = vars(args)
    pl.seed_everything(0)
    nba = NBADataModule(test_size=0.01, **dict_args)
    model = NBAResNet(**dict_args)
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
    print(datetime.now() - start)

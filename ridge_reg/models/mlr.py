from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adamax, AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from modules.core import (
    NBADataModule,
    NBAEncoder,
    ResBlock,
    NBASetEncoder,
    NBAEmbEncoder,
    SetEmbeddingBag,
    BiInteraction,
)
from models.ridge import NBARidge
from utils import load_nba, train_val_test_split


class NBAMixedLogit(NBASetEncoder):
    """
    mixed logistic regression
    """

    def __init__(
        self,
        # team_data_path="data/team_map.csv",
        # player_data_path="data/player_map.csv",
        lr=0.01,
        weight_decay=[0.0],
        team_emb_dim=2,
        player_emb_dim=2,
        **kwargs,
    ):
        """
        init method
        """
        super().__init__(
            lr, weight_decay, team_emb_dim=team_emb_dim, player_emb_dim=player_emb_dim
        )
        num_group = 9

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

        self.region = nn.Sequential(
            nn.Linear(self.in_dim, num_group), nn.Softmax(dim=-1)
        )
        self.made_p = nn.Linear(self.in_dim, num_group)
        self.fc = nn.Linear(num_group + self.in_dim, self.out_dim)

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


# class NBARidgeMLR(NBAMixedLogit):
#     """
#     mixed logistic regression and ridge regression ensemble
#     """

#     def __init__(self, lr, weight_decay, **kwargs):
#         super().__init__(lr=lr, weight_decay=weight_decay, **kwargs)
#         # self.fc0 = nn.Linear(2, self.out_dim)
#         # self.team = nn.Embedding(self.n_team, self.n_team_emb)
#         # self.player = nn.Embedding(self.n_player, self.n_player_emb)

#     def forward(self, x, return_embedding=True):
#         """
#         representations
#         """
#         fc, offt, deft, offp, offpa, defp, defpa = x
#         # fc = self.fc(fc)
#         offt = self.off_team(offt).view(-1, self.n_team_emb)
#         deft = self.def_team(deft).view(-1, self.n_team_emb)
#         offpa = self.off_player_age(offp, per_sample_weights=offpa)
#         defpa = self.def_player_age(defp, per_sample_weights=defpa)
#         offp = self.off_player(offp)
#         defp = self.def_player(defp)
#         ridge_emb = torch.cat([fc, offt, deft, offp, offpa, defp, defpa], dim=1)
#         off = torch.cat([offp, offpa], dim=1)
#         off = self.off(off)
#         off = F.silu(off)
#         offtp = torch.cat([offt, off], dim=1)
#         offtp = self.offtp(offtp)
#         offtp = F.celu(offtp)
#         deff = torch.cat([defp, defpa], dim=1)
#         deff = self.deff(deff)
#         deff = F.silu(deff)
#         deftp = torch.cat([deft, deff], dim=1)
#         deftp = self.deftp(deftp)
#         deftp = F.celu(deftp)
#         # mix = self.mix(offtp, deftp)

#         # offdef = self.mix(off, deff)
#         # offdef = F.selu(offdef)
#         # emb = torch.cat([fc, offt, deft, offp, offpa, defp, defpa], dim=1)
#         # emb = torch.cat([fc, offt, deft, off, deff], dim=1)
#         emb = torch.cat([fc, offtp, deftp], dim=1)
#         emb = self.resblock(emb)
#         emb = F.celu(emb)

#         if return_embedding:
#             return torch.cat([offt, deft, offp, offpa, defp, defpa], dim=1)
#         # return fc + offt + deft + offpa + defpa + offp + defp
#         region_dist = self.region(emb)
#         made_p = torch.sigmoid(self.made_p(emb))
#         result = region_dist * made_p * torch.linspace(0, 4, 9)
#         # made_p = self.made_p(emb)
#         # result = region_dist * made_p
#         # result = torch.sum(
#         #     region_dist * made_p, dim=-1
#         # )
#         result = torch.cat([ridge_emb, result], dim=1)
#         return self.fc(result)


class NBARidgeMLR(NBAMixedLogit):
    """
    mixed logistic regression and ridge regression ensemble
    points from 0 to 3
    """

    def __init__(self, lr, weight_decay, **kwargs):
        super().__init__(lr=lr, weight_decay=weight_decay, **kwargs)
        # self.fc0 = nn.Linear(2, self.out_dim)
        self.in_dim = 2 + 2 * self.n_player_emb + 2 * self.n_team_emb
        num_group = 4
        # offdef_dim = self.n_player_emb * 2
        tp_dim = self.n_team_emb + self.n_player_emb
        self.team = nn.Embedding(self.n_team, self.n_team_emb)
        self.player = SetEmbeddingBag(self.n_player, self.n_player_emb, mode="mean")
        self.off_player = SetEmbeddingBag(self.n_player, self.n_player_emb, mode="mean")
        self.def_player = SetEmbeddingBag(self.n_player, self.n_player_emb, mode="mean")
        self.off = nn.Linear(2 * self.n_player_emb, self.n_player_emb)
        self.deff = nn.Linear(2 * self.n_player_emb, self.n_player_emb)
        self.offtp = nn.Linear(tp_dim, tp_dim)
        self.deftp = nn.Linear(tp_dim, tp_dim)
        # self.mix = nn.Bilinear(tp_dim, tp_dim, tp_dim * 2)
        self.resblock = ResBlock(self.in_dim, self.in_dim)
        self.threshold = nn.Threshold(0, 0)
        # self.bn = nn.BatchNorm1d(self.in_dim)

        self.region = nn.Sequential(
            nn.Linear(self.in_dim, num_group), nn.Softmax(dim=-1)
        )
        self.made_p = nn.Linear(self.in_dim, num_group)
        # self.bi = BiInteraction()
        self.fc = nn.Linear(
            num_group + self.in_dim + 2 * self.n_player_emb, self.out_dim
        )

    def forward(self, x, return_embedding=True):
        """
        representations
        """
        fc, offt, deft, offp, offpa, defp, defpa = x
        # fc = self.fc(fc)
        offt = self.off_team(offt).view(-1, self.n_team_emb)
        deft = self.def_team(deft).view(-1, self.n_team_emb)
        offp = torch.cat([self.player(offp), self.off_player(offp)], dim=1)
        defp = torch.cat([self.player(defp), self.def_player(defp)], dim=1)
        ridge_emb = torch.cat([fc, offt, deft, offp, defp], dim=1)
        off = self.off(offp)
        off = F.silu(off)
        offtp = torch.cat([offt, off], dim=1)
        offtp = self.offtp(offtp)
        offtp = F.celu(offtp)
        deff = self.deff(defp)
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
        # emb = self.bn(emb)

        if return_embedding:
            return torch.cat([offt, deft, offp, defp], dim=1)
        # return fc + offt + deft + offpa + defpa + offp + defp
        emb = F.dropout(emb, p=0.1)
        region_dist = self.region(emb)
        made_p = torch.sigmoid(self.made_p(emb))
        result = region_dist * made_p * torch.linspace(0, 3, 4)
        # made_p = self.made_p(emb)
        # result = region_dist * made_p
        # result = torch.sum(
        #     region_dist * made_p, dim=-1
        # )
        result = torch.cat([ridge_emb, result], dim=1)
        return self.fc(result)


class NBAMLRShootEmb(NBAEmbEncoder):
    """
    mixed logistic regression and ridge regression ensemble and shooting
    self.offshoot represents shooting ability
    """

    def __init__(
        self,
        lr,
        weight_decay,
        team_emb_dim=128,
        player_emb_dim=128,
        num_heads=4,
        **kwargs,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            team_emb_dim=team_emb_dim,
            player_emb_dim=player_emb_dim,
            **kwargs,
        )
        self.in_dim = 2 + 4 * self.n_player_emb + 4 * self.n_team_emb
        num_group = 7
        # offdef_dim = self.n_player_emb * 2
        tp_dim = 2 * (self.n_team_emb + self.n_player_emb)

        self.team = nn.Embedding(self.n_team, self.n_team_emb)
        self.player = nn.Embedding(self.n_player, self.n_player_emb)
        self.transformer = nn.Transformer(
            d_model=2 * player_emb_dim,
            nhead=num_heads,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
        )
        self.tf_lr = nn.Sequential(
            nn.Linear(2 * self.n_player_emb, 2 * self.n_player_emb), nn.ReLU()
        )
        self.tf_bn = nn.BatchNorm1d(5)

        self.off = nn.Linear(2 * self.n_player_emb, 2 * self.n_player_emb)
        self.deff = nn.Linear(2 * self.n_player_emb, 2 * self.n_player_emb)

        self.offtp = nn.Linear(tp_dim, tp_dim)
        self.deftp = nn.Linear(tp_dim, tp_dim)

        # self.mix = nn.Bilinear(tp_dim, tp_dim, tp_dim * 2)
        self.resblock = ResBlock(self.in_dim, self.in_dim)
        # self.bn = nn.BatchNorm1d(self.in_dim)

        self.region = nn.Sequential(
            nn.Linear(self.in_dim, num_group), nn.Softmax(dim=-1)
        )
        self.made_p = nn.Linear(self.in_dim, num_group)
        self.fc = nn.Linear(
            num_group + 2 + 4 * self.n_team_emb + 4 * self.n_player_emb, self.out_dim
        )

    def transpose_transformer_pooling(self, query, key):
        query = torch.transpose(query, 0, 1)
        key = torch.transpose(key, 0, 1)
        out = self.transformer(query, key)
        out = torch.transpose(out, 0, 1)
        res = self.tf_lr(out)
        res = res + out
        res = self.tf_bn(res)
        return torch.max(out, dim=1)

    def forward(self, x, return_embedding=True):
        """
        representations
        """
        fc, offt, deft, offp, offpa, defp, defpa = x
        # fc = self.fc(fc)
        offt = torch.cat(
            [
                self.team(offt).view(offt.size(0), -1),
                self.off_team(offt).view(-1, self.n_team_emb),
            ],
            dim=-1,
        )
        deft = torch.cat(
            [
                self.team(deft).view(deft.size(0), -1),
                self.def_team(deft).view(-1, self.n_team_emb),
            ],
            dim=-1,
        )
        offp = torch.cat([self.player(offp), self.off_player(offp)], dim=-1)
        defp = torch.cat([self.player(defp), self.def_player(defp)], dim=-1)

        # transformer for shooting
        out = self.transpose_transformer_pooling(defp, offp)

        # ridge regression
        offp = torch.mean(offp, dim=1)
        defp = torch.mean(defp, dim=1)
        ridge_emb = torch.cat([fc, offt, deft, offp, defp], dim=1)

        # mixed logistic
        off = self.off(offp)
        off = F.silu(off)
        offtp = torch.cat([offt, off], dim=1)
        offtp = self.offtp(offtp)
        offtp = F.celu(offtp)
        deff = self.deff(defp)
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
        # emb = self.bn(emb)

        if return_embedding:
            return torch.cat([offt, deft, offp, defp], dim=1)
        # return fc + offt + deft + offpa + defpa + offp + defp
        emb = F.dropout(emb, p=0.1)
        region_dist = self.region(emb)
        made_p = torch.sigmoid(self.made_p(emb))
        result = region_dist * made_p * torch.linspace(0, 3, 7)
        # made_p = self.made_p(emb)
        # result = region_dist * made_p
        # result = torch.sum(
        #     region_dist * made_p, dim=-1
        # )
        result = torch.cat([ridge_emb, result], dim=1)
        return self.fc(result)


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
    model = NBAMixedLogit(**dict_args)
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

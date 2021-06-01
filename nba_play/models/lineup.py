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
)
from modules.set_transformer import PMA
from models.ridge import NBARidge
from utils import load_nba, train_val_test_split


class NBAReformer(NBAEncoder):
    """NBA transformer"""

    def __init__(
        self,
        lr,
        weight_decay,
        n_team=30,
        team_emb_dim=64,
        n_player=531,
        player_emb_dim=64,
        num_heads=4,
        **kwargs
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            n_team=n_team,
            team_emb_dim=team_emb_dim,
            n_player=n_player,
            player_emb_dim=player_emb_dim,
            **kwargs,
        )
        self.in_dim = 2 + self.n_player_emb + 2 * self.n_team_emb
        self.fc = nn.Linear(self.in_dim, self.out_dim)
        self.player = nn.Embedding(n_player, player_emb_dim)
        self.off_player = nn.Embedding(n_player, player_emb_dim)
        self.def_player = nn.Embedding(n_player, player_emb_dim)
        # self.off_attn = nn.MultiheadAttention(player_emb_dim, num_heads)
        # self.def_attn = nn.MultiheadAttention(player_emb_dim, num_heads)
        # self.out_attn = nn.MultiheadAttention(player_emb_dim, num_heads)
        self.transformer = nn.Transformer(
            d_model=player_emb_dim,
            nhead=num_heads,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
        )

    def forward(self, x, return_embedding=True):
        fc, offt, deft, offp, offpa, defp, defpa = x
        offt = self.off_team(offt).squeeze()
        deft = self.def_team(deft).squeeze()
        offp = torch.cat([self.player(offp), self.off_player(offp)], dim=1)
        defp = torch.cat([self.player(defp), self.off_player(defp)], dim=1)
        # offp = self.off_player(offp)
        # defp = self.def_player(defp)
        offp = torch.transpose(offp, 0, 1)
        defp = torch.transpose(defp, 0, 1)
        # offp, _ = self.off_attn(offp, offp, offp)
        # defp, _ = self.def_attn(defp, defp, defp)
        # out, _ = self.out_attn(defp, offp, offp)
        out = self.transformer(defp, offp)
        out = torch.transpose(out, 0, 1)
        out = torch.mean(out, dim=1)
        out = torch.cat([fc, offt, deft, out], dim=1)
        return self.fc(out)


class NBAShootTF(NBAEmbEncoder):
    """
    mixed logistic regression and ridge regression ensemble and shooting
    self.offshoot represents shooting ability
    """

    def __init__(
        self,
        lr,
        weight_decay,
        team_emb_dim=32,
        player_emb_dim=32,
        num_heads=4,
        **kwargs
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
        # self.transformer = nn.Transformer(
        #     d_model=2 * player_emb_dim,
        #     nhead=num_heads,
        #     num_encoder_layers=1,
        #     num_decoder_layers=1,
        #     dim_feedforward=64,
        # )
        d_model = 2 * player_emb_dim
        self.off_encoder = nn.TransformerEncoderLayer(d_model, num_heads, 64)
        self.def_encoder = nn.TransformerEncoderLayer(d_model, num_heads, 64)
        self.pooling = PMA(d_model, num_heads, 1)
        self.decoder = nn.TransformerDecoderLayer(d_model, num_heads, 64)
        self.tf_lr = nn.Sequential(
            nn.Linear(2 * self.n_player_emb, 2 * self.n_player_emb), nn.ReLU()
        )
        # self.tf_bn = nn.BatchNorm1d(5)

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
        # self.fc = nn.Linear(
        #     num_group + 2 + 4 * self.n_team_emb + 4 * self.n_player_emb, self.out_dim
        # )

    def transpose_transformer_pooling(self, query, key):
        query = torch.transpose(query, 0, 1)
        key = torch.transpose(key, 0, 1)
        # out = self.transformer(query, key)
        query = self.def_encoder(query)
        key = self.off_encoder(key)
        query = torch.transpose(query, 0, 1)
        query = self.pooling(query)
        query = torch.transpose(query, 0, 1)
        out = self.decoder(query, key)
        out = torch.transpose(out, 0, 1)
        res = self.tf_lr(out)
        res = res + out
        # res = self.tf_bn(res)
        return torch.amax(out, dim=1)

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
        result = torch.cat([ridge_emb, result, out], dim=1)
        return self.fc(result)

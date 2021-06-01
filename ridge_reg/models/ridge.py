import os
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

from modules.core import NBADataModule, NBAEncoder
from utils import load_nba, train_val_test_split


class NBARidge(NBAEncoder):
    """
    NBA Representation Learning
    """

    def __init__(
        self, lr=0.01, weight_decay=0, team_emb_dim=1, player_emb_dim=1, **kwargs,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            team_emb_dim=team_emb_dim,
            player_emb_dim=player_emb_dim,
            **kwargs,
        )
        self.fc0 = nn.Linear(2, self.out_dim)

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
        fc = self.fc0(fc)
        offt = self.off_team(offt).view(-1, self.n_team_emb)
        deft = self.def_team(deft).view(-1, self.n_team_emb)
        # offpa = self.off_player_age(offp, per_sample_weights=offpa)
        # defpa = self.def_player_age(defp, per_sample_weights=defpa)
        offp = self.off_player(offp)
        defp = self.def_player(defp)

        # team representations
        # off = torch.cat([offp, offpa], dim=1)
        # off = self.off(off)
        # off = F.silu(off)
        # offtp = torch.cat([offt, off], dim=1)
        # offtp = self.offtp(offtp)
        # offtp = F.celu(offtp)
        # deff = torch.cat([defp, defpa], dim=1)
        # deff = self.deff(deff)
        # deff = F.silu(deff)
        # deftp = torch.cat([deft, deff], dim=1)
        # deftp = self.deftp(deftp)
        # deftp = F.celu(deftp)
        # emb = torch.cat([fc, offtp, deftp], dim=1)
        # emb = self.resblock(emb)
        # emb = F.celu(emb)
        # mix = self.mix(offtp, deftp)

        # offdef = self.mix(off, deff)
        # offdef = F.selu(offdef)
        emb = torch.cat([fc, offt, deft, offp, defp], dim=1)
        # emb = torch.cat([fc, offt, deft, offp, offpa, defp, defpa], dim=1)
        # emb = torch.cat([fc, offt, deft, off, deff], dim=1)

        if return_embedding:
            # return torch.cat([offt, deft, offp, offpa, defp, defpa], dim=1)
            return emb
        return torch.sum(emb, dim=1)
        # return self.threshold(self.fc(emb))

    # def training_step(self, batch, batch_idx, optimizer_idx):
    def training_step(self, batch, batch_idx):
        """
        training step
        """
        loss = self.mse_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def training_epoch_end(self, training_step_outputs):
        """
        training epoch end
        """
        logger = self.logger.experiment
        logger.add_histogram("fc_weight", self.fc0.weight, self.current_epoch)

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
        opt_param = [
            {"params": self.fc0.parameters(), "weight_decay": 0},
            {"params": self.off_team.parameters(), "weight_decay": self.wd[0]},
            {"params": self.def_team.parameters(), "weight_decay": self.wd[1]},
            {"params": self.off_player.parameters(), "weight_decay": self.wd[2]},
            {"params": self.off_player_age.parameters(), "weight_decay": self.wd[3],},
            {"params": self.def_player.parameters(), "weight_decay": self.wd[4]},
            {"params": self.def_player_age.parameters(), "weight_decay": self.wd[5]},
        ]
        # dense_opt = AdamW(dense_param, lr=self.lr)
        # sparse_opt = Adamax(sparse_param, lr=self.lr)
        # dense_scheduler = OneCycleLR(dense_opt, max_lr=0.1, total_steps=1000)
        # sparse_scheduler = OneCycleLR(sparse_opt, max_lr=0.1, total_steps=1000)
        # return [dense_opt, sparse_opt], [dense_scheduler, sparse_scheduler]
        return AdamW(opt_param, lr=self.lr)

    # def mse_loss(self, batch):
    #     """
    #     calculate mse loss
    #     """
    #     x, y = batch
    #     yhat = self(x, return_embedding=False)
    #     return F.mse_loss(torch.flatten(yhat), torch.flatten(y))

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        model specific args
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=0.05)
        # parser.add_argument("--team_data_path", type=str, default="data/team_map.csv")
        # parser.add_argument(
        #     "--player_data_path", type=str, default="data/player_map.csv"
        # )
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


def extract_player_embeddings(ckpt_path):
    """
    extract player embeddings
    """
    model = NBARidge.load_from_checkpoint(checkpoint_path=ckpt_path)
    offp = model.off_player.weight.detach().numpy()
    defp = model.def_player.weight.detach().numpy()
    offpa = model.off_player_age.weight.detach().numpy()
    defpa = model.def_player_age.weight.detach().numpy()
    return np.concatenate([offp, defp, offpa, defpa], axis=1)


def save_output(model, file_name):
    x, _ = load_nba(path="data/nba_2018/nba_2018_test.csv", to_tensor=True)
    model.freeze()
    yhat = model(x, return_embedding=False)
    if file_name:
        np.save(f"data/output/{file_name}.npy", yhat.numpy().squeeze())


# tmp = None
# tmp1 = DataLoader(NBADataset(tmp), batch_size=3)
# tmp2, tmp3 = next(iter(tmp1))
# tmp2[3]
# tmpp = NBAEncoderModule()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out_file", type=str, default="")
    parser = NBADataModule.add_data_specific_args(parser)
    parser = NBARidge.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    nba_early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min"
    )
    dict_args = vars(args)
    pl.seed_everything(0)
    nba = NBADataModule(**dict_args)
    model = NBARidge(**dict_args)
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
    os.system('say "Your Python Program has Finished"')

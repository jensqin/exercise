from argparse import ArgumentParser
import time
import pickle

import numpy as np
from numpy.lib.npyio import load
import pandas as pd
import pyro
import pytorch_lightning as pl
import torch
from pyro import distributions as dist
from pyro.distributions import constraints
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoLowRankMultivariateNormal
from pyro.nn import PyroModule, PyroParam, PyroSample
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn import metrics
from torch import nn
from torch.functional import split
from torch.nn import functional as F
from torch.optim import Adamax, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.dataloader import DataLoader

from nba_torch import NBADataModule, NBADataset, NBAEncoder
from utils import load_nba, summary_samples, transform_to_array


class NBABayesEncoderModule(pl.LightningModule):
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


class NBABayesEncoder(PyroModule):
    """
    NBA hierachical bayesian encoder
    """

    def __init__(self, n_team=30, n_player=531, n_group=7, scale_global=1):
        super().__init__()
        # n_team, n_player = NBAEncoder.n_team_and_player(
        #     team_data_path, player_data_path
        # )
        # n_team, n_player = 5, 85
        n_team_emb, n_player_emb = 1, 1
        # self.lr = lr
        # self.wd = weight_decay
        scale_icept = 10
        # scale_icept, scale_global = hparam
        local_vec = torch.ones(n_group)
        tau = pyro.sample("tau", dist.HalfCauchy(scale_global))
        lam = pyro.sample("lambda", dist.HalfCauchy(local_vec).to_event())
        # sig = (lam ** 2) * (tau ** 2)
        self.sig = tau * lam
        # ws = pyro.sample("ws", dist.Normal(0, sig).to_event())
        self.fc = PyroModule[nn.Linear](2, 1)
        self.fc.weight = PyroSample(dist.Normal(0.0, 1.0).expand([1, 2]).to_event(2))
        self.fc.bias = PyroSample(
            dist.StudentT(4, 0.0, scale_icept).expand([1]).to_event(1)
        )
        # self.off_team = PyroModule[nn.Embedding](n_team, n_team_emb)
        # self.off_team.weight = PyroSample(
        #     dist.Normal(0.0, 1.0).expand([n_team, 1]).to_event(1)
        # )
        # self.def_team = PyroModule[nn.Embedding](n_team, n_team_emb)
        # self.def_team.weight = PyroSample(
        #     dist.Normal(0.0, 1.0).expand([n_team, 1]).to_event(1)
        # )
        # self.off_player = PyroModule[nn.EmbeddingBag](
        #     n_player, n_player_emb, mode="sum"
        # )
        # self.off_player.weight = PyroSample(
        #     dist.Normal(0.0, 1.0).expand([n_player, 1]).to_event(1)
        # )
        # self.off_player_age = PyroModule[nn.EmbeddingBag](
        #     n_player, n_player_emb, mode="sum"
        # )
        # self.off_player.weight = PyroSample(
        #     dist.Normal(0.0, 1.0).expand([n_player, 1]).to_event(1)
        # )
        # self.def_player = PyroModule[nn.EmbeddingBag](
        #     n_player, n_player_emb, mode="sum"
        # )
        # self.def_player.weight = PyroSample(
        #     dist.Normal(0.0, 1.0).expand([n_player, 1]).to_event(1)
        # )
        # self.def_player_age = PyroModule[nn.EmbeddingBag](
        #     n_player, n_player_emb, mode="sum"
        # )
        # self.def_player_age.weight = PyroSample(
        #     dist.Normal(0.0, 1.0).expand([n_player, 1]).to_event(1)
        # )
        # player_dim = n_player_emb * 4
        # self.bn_player = nn.BatchNorm1d(player_dim)
        # self.fin = nn.Linear(2 + n_team_emb * 2 + n_player_emb * 4, 3)
        # self.sparse = nn.ModuleList(
        #     [self.off_player, self.off_player_age, self.def_player, self.def_player_age]
        # )

    def forward(self, x, y=None):
        """
        forward method
        """
        fc, offt, deft, offp, offpa, defp, defpa = x
        fc = self.fc(fc)
        # offt = self.off_team(offt).view(-1, 1)
        # deft = self.def_team(deft).view(-1, 1)
        # offpa = self.off_player_age(offp, per_sample_weights=offpa)
        # defpa = self.def_player_age(defp, per_sample_weights=defpa)
        # offp = self.off_player(offp)
        # defp = self.def_player(defp)
        mean = (
            fc * self.sig[0]
            # + offt * self.sig[1]
            # + deft * self.sig[2]
            # + offpa * self.sig[3]
            # + defpa * self.sig[4]
            # + offp * self.sig[5]
            # + defp * self.sig[6]
        )
        mean = torch.flatten(mean)
        sigma = pyro.sample("sigma", dist.HalfCauchy(1.0))
        with pyro.plate("data", fc.size(0)):
            _ = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean


class NBAGuide(PyroModule):
    """
    nba guide function
    """

    def __init__(self, n_input=7):
        super().__init__()
        self.sigma_loc = PyroParam(
            torch.tensor(1.0), constraint=constraints.interval(0.0, 10.0)
        )
        # We can be Bayesian about the linear parts.
        self.weight_loc = PyroParam(torch.zeros(1, n_input))
        self.weight_scale = PyroParam(
            torch.ones(1, n_input), constraint=constraints.positive
        )
        self.bias_loc = PyroParam(torch.zeros(1))
        self.bias_scale = PyroParam(torch.ones(1), constraint=constraints.positive)
        self.off_team_loc = PyroParam(torch.zeros(1, n_input))
        self.off_team_scale = PyroParam(
            torch.ones(1, n_input), constraint=constraints.positive
        )
        self.def_team_loc = PyroParam(torch.zeros(1, n_input))
        self.def_team_scale = PyroParam(
            torch.ones(1, n_input), constraint=constraints.positive
        )
        self.off_player_loc = PyroParam(torch.zeros(1, n_input))
        self.off_player_scale = PyroParam(
            torch.ones(1, n_input), constraint=constraints.positive
        )
        self.def_player_loc = PyroParam(torch.zeros(1, n_input))
        self.def_player_scale = PyroParam(
            torch.ones(1, n_input), constraint=constraints.positive
        )
        self.off_player_age_loc = PyroParam(torch.zeros(1, n_input))
        self.off_player__age_scale = PyroParam(
            torch.ones(1, n_input), constraint=constraints.positive
        )
        self.def_player_age_loc = PyroParam(torch.zeros(1, n_input))
        self.def_player_age_scale = PyroParam(
            torch.ones(1, n_input), constraint=constraints.positive
        )

    def forward(self, x, y=None):
        pyro.sample("sigma", dist.Delta(self.sigma_loc))
        pyro.sample(
            "fc.weight", dist.Normal(self.weight_loc, self.weight_scale).to_event(2)
        )
        pyro.sample("fc.bias", dist.Normal(self.bias_loc, self.bias_scale).to_event(1))
        pyro.sample(
            "fc.weight", dist.Normal(self.weight_loc, self.weight_scale).to_event(2)
        )
        pyro.sample(
            "off_team.weight",
            dist.Normal(self.weight_loc, self.weight_scale).to_event(2),
        )
        pyro.sample(
            "def_team.weight",
            dist.Normal(self.weight_loc, self.weight_scale).to_event(2),
        )
        pyro.sample(
            "off_player.weight",
            dist.Normal(self.weight_loc, self.weight_scale).to_event(2),
        )
        pyro.sample(
            "def_player.weight",
            dist.Normal(self.weight_loc, self.weight_scale).to_event(2),
        )
        pyro.sample(
            "off_player_age",
            dist.Normal(self.weight_loc, self.weight_scale).to_event(2),
        )
        pyro.sample(
            "def_player_age",
            dist.Normal(self.weight_loc, self.weight_scale).to_event(2),
        )


def val_mse(predictive, x, y):
    """
    validation mse
    """
    samples = predictive(x)
    pred_summary = summary_samples(samples)
    mu = pred_summary["_RETURN"]
    return metrics.mean_squared_error(mu["mean"], y)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    model = NBABayesEncoder()
    # guide = AutoDiagonalNormal(model)
    guide = AutoLowRankMultivariateNormal(model)
    # datam = NBADataModule()
    # train_loader = datam.train_dataloader()
    # tmp = iter(train_loader)

    # float_cols = ["y", "HomeAway", "ScoreDiff"] + [f"age{x}" for x in range(1, 11)]
    # type_dict = {key: np.float32 for key in float_cols}
    # dfset = NBADataset(pd.read_csv("data/nba_nw.csv", dtype=type_dict))
    # loader = DataLoader(dfset, batch_size=32)
    # tmp = iter(loader)
    # train, test = load_nba(split_mode="test", test=0.2)
    train = load_nba(path="data/nba_2018/nba_2018_train.csv")
    val = load_nba(path="data/nba_2018/nba_2018_test.csv")
    test = val
    # test = load_nba(path="data/nba_2018/test.csv")
    train_loader = DataLoader(NBADataset(train), batch_size=32)

    adam = pyro.optim.Adamax({"lr": args.lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    X_train, y_train = transform_to_array(val)
    X_val, y_val = transform_to_array(val)
    X_test, y_test = transform_to_array(test)
    predictive = Predictive(
        model,
        guide=guide,
        num_samples=1000,
        return_sites=("fc.weight", "obs", "_RETURN"),
    )

    pyro.clear_param_store()
    # num_epochs = 3
    start = time.time()
    # early stopping
    patience = 5
    mse_benchmark = np.zeros(patience)
    for i in range(args.epochs):
        loss = 0
        for batch in iter(train_loader):
            X, y = batch
            y = torch.flatten(y)
            loss = svi.step(X, y)
            loss /= len(y)
        train_mse = val_mse(predictive, X_train, y_train)
        ith_mse = val_mse(predictive, X_val, y_val)
        if i % 1 == 0:
            print(
                f"epoch {i + 1}: train loss {loss}, train mse {train_mse}, val mse {ith_mse}"
            )
            if i > patience and ith_mse > mse_benchmark.max():
                print(f"Early stopping at {i}th epoch. MSE: {ith_mse}.")
                break
            mse_benchmark[i % patience] = ith_mse
    print(time.time() - start)

    samples = predictive(X_test)
    pred_summary = summary_samples(samples)
    with open("data/samples/svi.pkl", "wb") as f:
        pickle.dump(pred_summary, f)
    mu = pred_summary["_RETURN"]
    yhat = pred_summary["obs"]
    # predictions = pd.DataFrame(
    #     {
    #         "mu_mean": mu["mean"],
    #         "mu_perc_5": mu["5%"],
    #         "mu_perc_95": mu["95%"],
    #         "y_mean": yhat["mean"],
    #         "y_perc_5": yhat["5%"],
    #         "y_perc_95": yhat["95%"],
    #         "true_y": y,
    #     }
    # )
    mse = metrics.mean_squared_error(mu["mean"], y_test)
    print(f"Test MSE: {mse}")

# python nba_pyro.py --lr 0.001 --epochs 30
# 146.47113513946533
# Test MSE: 1.3741856813430786

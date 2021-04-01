from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adamax, AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from utils import load_nba, train_val_test_split


class NBADataset(Dataset):
    """
    NBA NW Data
    """

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        # print(self.df.dtypes)
        self.idx = df.index
        x_cols = [
            ["HomeAway", "ScoreDiff"],
            ["OffTeam"],
            ["DefTeam"],
            ["P1", "P2", "P3", "P4", "P5"],
            ["age1", "age2", "age3", "age4", "age5"],
            ["P6", "P7", "P8", "P9", "P10"],
            ["age6", "age7", "age8", "age9", "age10"],
        ]
        y_cols = ["y"]
        self.x = [self.pd_to_tensor(self.df[col]) for col in x_cols]
        self.y = self.pd_to_tensor(self.df[y_cols])

    def __len__(self):
        return len(self.idx)

    @staticmethod
    def pd_to_tensor(x):
        xarray = x.values
        return torch.from_numpy(xarray)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # elif isinstance(idx, int):
        #     idx = [idx]
        sample_x = [t[idx, :] for t in self.x]
        sample_y = self.y[idx, :]
        return sample_x, sample_y


class NBABetDataset(NBADataset):
    """
    NBA 2018 Data
    """

    def __init__(self, df):
        super().__init__(df)
        yhat_cols = ["y_exp"]
        self.yhat = self.pd_to_tensor(self.df[yhat_cols])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_x = [t[idx, :] for t in self.x]
        sample_y = self.y[idx, :]
        sample_yhat = self.yhat[idx, :]
        return sample_x, sample_y, sample_yhat


class BetLoss(nn.Module):
    """bet loss"""

    def __init__(self):
        super().__init__()


class NBADataModule(pl.LightningDataModule):
    """
    NBA NW Division data
    """

    def __init__(
        self,
        # data_path="data/nba_nw.csv",
        num_workders=1,
        batch_size=32,
        val_size=0.15,
        test_size=0.1,
        betloss=True,
        **kwargs,
    ):
        super().__init__()
        # self.data_path = os.path.join(os.getcwd(), data_path)
        self.num_workders = num_workders
        self.batch_size = batch_size
        stratify_cols = ["OffTeam", "DefTeam"]
        float_cols = ["y", "HomeAway", "ScoreDiff"] + [f"age{x}" for x in range(1, 11)]
        # int_cols = stratify_cols + [f"P{x}" for x in range(1, 11)]
        type_dict = {key: np.float32 for key in float_cols}
        # type_dict.update({key: np.int16 for key in int_cols})
        # type_dict.update({"y": np.int32})
        # self.nba = pd.read_csv(data_path, dtype=type_dict)
        # train, val, test = train_val_test_split(
        #     self.nba,
        #     val=val_size,
        #     test=test_size,
        #     shuffle=True,
        #     stratify_cols=stratify_cols,
        #     random_state=None,
        # )
        train = load_nba(path="data/nba_2018/nba_2018_train.csv")
        val = load_nba(path="data/nba_2018/nba_2018_test.csv")
        test = val
        self.betloss = betloss
        if betloss:
            self.train = NBABetDataset(train)
            self.val = NBABetDataset(val)
            self.test = NBABetDataset(test)
        else:
            self.train = NBADataset(train)
            self.val = NBADataset(val)
            self.test = NBADataset(test)

    def setup(self, stage=None):
        pass

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


class SetEmbedding(nn.Module):
    """
    set embedding
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        maxp = torch.amax(x, dim=1, keepdim=True)
        combo = self.fc1(x) + self.fc2(maxp)
        return torch.sigmoid(combo)


class SetEmbeddingBag(nn.Module):
    """
    set embedding
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, mode: str = "mean"):
        super().__init__()
        self.mode = mode
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.en_fc = nn.Linear(embedding_dim, embedding_dim)
        self.de_fc = nn.Linear(embedding_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def encoder(self, x):
        x = self.en_fc(x)
        return F.celu(x)

    def decoder(self, x):
        x = self.de_fc(x)
        return F.celu(x)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        maxp = torch.amax(x, dim=1, keepdim=True)
        combo = self.fc1(x) + self.fc2(maxp)
        if self.mode == "mean":
            combo = torch.mean(combo, dim=1)
        elif self.mode == "max":
            combo = torch.amax(combo, dim=1)
        elif self.mode == "sum":
            combo = torch.sum(combo, dim=1)
        else:
            raise NotImplementedError(f"{self.mode} mode is not implemented.")
        return self.decoder(combo)


class ResBlock(nn.Module):
    """
    resnet block
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # res = self.bn1(x)
        # res = F.relu(res)
        # res = self.fc1(res)
        # res = self.bn2(res)
        # res = F.relu(res)
        # res = self.fc2(res)
        res = self.fc1(x)
        res = self.bn1(res)
        res = F.relu(res)
        res = self.fc2(res)
        # res = self.bn2(res)
        # res = F.relu(res)
        return x + res


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
        n_team=30,
        team_emb_dim=1,
        n_player=531,
        player_emb_dim=1,
        betloss=True,
        **kwargs,
    ):
        super().__init__()
        # n_team, n_player = NBAEncoder.n_team_and_player(
        #     team_data_path, player_data_path
        # )
        n_team, n_player = 30, 531
        self.betloss = betloss
        self.n_team_emb, self.n_player_emb = team_emb_dim, player_emb_dim
        self.lr = lr
        self.wd = weight_decay
        self.in_dim = 2 + 4 * self.n_player_emb + 2 * self.n_team_emb
        self.out_dim = 1
        self.fc = nn.Linear(self.in_dim, self.out_dim)
        self.off_team = nn.Embedding(n_team, self.n_team_emb)
        self.def_team = nn.Embedding(n_team, self.n_team_emb)
        # self.dense = nn.ModuleList([self.fc, self.off_team, self.def_team])
        self.off_player = nn.EmbeddingBag(n_player, self.n_player_emb, mode="mean")
        # self.off_player = nn.Sequential(
        #     nn.Embedding(n_player, self.n_player_emb),
        #     SetTransformer(self.n_player_emb, 1, self.n_player_emb, num_heads=4),
        # )
        self.off_player_age = nn.EmbeddingBag(n_player, self.n_player_emb, mode="sum")
        self.def_player = nn.EmbeddingBag(n_player, self.n_player_emb, mode="mean")
        # self.def_player = nn.Sequential(
        #     nn.Embedding(n_player, self.n_player_emb),
        #     SetTransformer(self.n_player_emb, 1, self.n_player_emb, num_heads=4),
        # )
        self.def_player_age = nn.EmbeddingBag(n_player, self.n_player_emb, mode="sum")

        # resnet
        # offdef_dim = self.n_player_emb * 2
        # tp_dim = self.n_team_emb + offdef_dim
        # self.off = nn.Linear(self.n_player_emb * 2, self.n_player_emb * 2)
        # self.deff = nn.Linear(self.n_player_emb * 2, self.n_player_emb * 2)
        # self.offtp = nn.Linear(tp_dim, tp_dim)
        # self.deftp = nn.Linear(tp_dim, tp_dim)
        # # self.mix = nn.Bilinear(tp_dim, tp_dim, tp_dim * 2)
        # self.resblock = ResBlock(self.in_dim, self.in_dim)
        # self.threshold = nn.Threshold(0, 0)
        # # self.fc1 = nn.Linear(in_dim, in_dim)
        # # player_dim = n_player_emb * 4
        # # self.bn_player = nn.BatchNorm1d(player_dim)
        # # self.fin = nn.Linear(2 + n_team_emb * 2 + n_player_emb * 4, 3)
        # # self.sparse = nn.ModuleList(
        # #     [self.off_player, self.off_player_age, self.def_player, self.def_player_age]
        # # )

    def forward(self, x):
        raise NotImplementedError

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
        opt_param = [
            {"params": self.fc.parameters(), "weight_decay": 0},
            {"params": self.off_team.parameters(), "weight_decay": self.wd[0]},
            {"params": self.def_team.parameters(), "weight_decay": self.wd[1]},
            {"params": self.off_player.parameters(), "weight_decay": self.wd[2]},
            {"params": self.off_player_age.parameters(), "weight_decay": self.wd[3],},
            {"params": self.def_player.parameters(), "weight_decay": self.wd[4]},
            {"params": self.def_player_age.parameters(), "weight_decay": self.wd[5]},
        ]
        # dense_param = [
        #     {"params": self.fc.parameters(), "weight_decay": 0},
        #     {"params": self.off_team.parameters(), "weight_decay": self.wd[0]},
        #     {"params": self.def_team.parameters(), "weight_decay": self.wd[1]},
        # ]
        # sparse_param = [
        #     {"params": self.off_player.parameters(), "weight_decay": self.wd[2]},
        #     {"params": self.off_player_age.parameters(), "weight_decay": self.wd[3]},
        #     {"params": self.def_player.parameters(), "weight_decay": self.wd[4]},
        #     {"params": self.def_player_age.parameters(), "weight_decay": self.wd[5]},
        # ]
        optimizer = AdamW(opt_param, lr=self.lr)
        # dense_opt = AdamW(dense_param, lr=self.lr)
        # sparse_opt = Adamax(sparse_param, lr=self.lr)
        # dense_scheduler = OneCycleLR(dense_opt, max_lr=0.1, total_steps=1000)
        # sparse_scheduler = OneCycleLR(sparse_opt, max_lr=0.1, total_steps=1000)
        # return [dense_opt, sparse_opt], [dense_scheduler, sparse_scheduler]
        return optimizer

    def mse_loss(self, batch):
        """
        calculate mse loss
        """
        if self.betloss:
            x, y, y_exp = batch
            yhat = torch.flatten(self(x, return_embedding=False))
            y, y_exp = torch.flatten(y), torch.flatten(y_exp)
            ysign = torch.sign(y_exp - y)
            bet_result = torch.sigmoid(100 * (yhat - y_exp)) * ysign
            return torch.mean(bet_result)
        else:
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


def extract_player_embeddings(model_cls, ckpt_path):
    """
    extract player embeddings
    """
    model = model_cls.load_from_checkpoint(checkpoint_path=ckpt_path)
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
        path = f"data/output/{file_name}.npy"
        np.save(path, yhat.numpy().squeeze())
        print(f"Saved file to {path}.")
    else:
        print("File name is None, no outputs are saved.")


class NBASetEncoder(NBAEncoder):
    """
    NBA Deep Set Encoder
    """

    def __init__(
        self,
        lr=0.01,
        weight_decay=0,
        n_team=30,
        team_emb_dim=1,
        n_player=531,
        player_emb_dim=1,
        **kwargs,
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
        self.n_player = n_player
        self.n_team = n_team
        self.off_player = SetEmbeddingBag(n_player, self.n_player_emb, mode="mean")
        self.def_player = SetEmbeddingBag(n_player, self.n_player_emb, mode="mean")


class NBAEmbEncoder(NBAEncoder):
    """
    NBA Player Embedding Encoder
    """

    def __init__(
        self,
        lr=0.01,
        weight_decay=0,
        n_team=30,
        team_emb_dim=1,
        n_player=531,
        player_emb_dim=1,
        **kwargs,
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
        self.n_player = n_player
        self.n_team = n_team
        self.off_player = nn.Embedding(n_player, self.n_player_emb)
        self.def_player = nn.Embedding(n_player, self.n_player_emb)


class FM(nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
    """

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


class BiInteraction(nn.Module):
    """Bi-Interaction Layer
     pairwise element-wise product of features into one single vector.

      Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embedding_size)``.

      Output shape
        - 3D tensor with shape: ``(batch_size,1,embedding_size)``.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        concated_embeds_value = inputs
        square_of_sum = torch.pow(
            torch.sum(concated_embeds_value, dim=1, keepdim=True), 2
        )
        sum_of_square = torch.sum(
            concated_embeds_value * concated_embeds_value, dim=1, keepdim=True
        )
        return 0.5 * (square_of_sum - sum_of_square)


class NBAInteractionEncoder(NBAEncoder):
    """
    NBA Player Embedding Encoder
    """

    def __init__(
        self,
        lr=0.01,
        weight_decay=0,
        n_team=30,
        team_emb_dim=1,
        n_player=531,
        player_emb_dim=1,
        **kwargs,
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
        self.n_player = n_player
        self.n_team = n_team
        self.off_player = nn.Sequential(
            nn.Embedding(n_player, self.n_player_emb), BiInteraction()
        )
        self.def_player = nn.Sequential(
            nn.Embedding(n_player, self.n_player_emb), BiInteraction()
        )

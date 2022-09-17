import itertools
import math
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import torch
from torch import nn
from torch._C import long
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from ray import tune

# from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

from boston.boston_ts import X, y

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
X_ts = torch.as_tensor(Xs, dtype=torch.float)
y_ts = torch.as_tensor(y, dtype=torch.float).view(-1, 1)


class GroupRidge(nn.Module):
    """
    group ridge regression
    """

    def __init__(self, in_features, out_features, group=None, bias=True):
        """
        init method
        """
        super(GroupRidge, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group = group
        if group is None:
            self.group_weight = nn.ParameterList(
                [
                    nn.parameter.Parameter(torch.Tensor(out_features, 1))
                    for _ in range(in_features)
                ]
            )
        else:
            group_size = self._parse_group()
            self.group_weight = nn.ParameterList(
                [
                    nn.parameter.Parameter(torch.Tensor(out_features, x))
                    for x in group_size
                ]
            )
        self.weight = torch.cat(list(self.group_weight), dim=1)
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def _parse_group(self):
        """
        parse group argument
        """
        group_size = [len(x) for x in self.group]
        assert self.in_features == sum(
            group_size
        ), "Input dimension and group size mismatch!"
        elements = itertools.chain.from_iterable(self.group)
        assert all(
            0 <= x < self.in_features for x in elements
        ), "Elements of group MUST be less than or equal to in_features."
        return group_size

    def reset_parameters(self) -> None:
        for param in self.group_weight:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.weight = torch.cat(list(self.group_weight), dim=1)
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


# ridge = GroupRidge(13, 1)
# mse_critirion = nn.MSELoss()
# opt_param = [
#     {"params": [ridge.group_weight[x]], "weight_decay": 0.1} for x in range(13)
# ]
# opt_param += [{"params": ridge.bias, "weight_decay": 0}]
# optimizer = SGD(opt_param, lr=0.05)


class RegTrainable(tune.Trainable):
    def setup(self, config):
        # config (dict): A dict of hyperparameters
        self.ridge = GroupRidge(13, 1)
        self.mse_critirion = nn.MSELoss()
        opt_param = [
            {"params": [item], "weight_decay": config[f"weight_decay_{id}"],}
            for id, item in enumerate(self.ridge.group_weight)
        ]
        opt_param += [{"params": self.ridge.bias, "weight_decay": 0}]
        self.optimizer = SGD(opt_param, lr=config["lr"])

    def step(self):  # This is called iteratively.
        for _ in range(20):
            out = self.ridge(X_ts)
            loss = self.mse_critirion(out, y_ts)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        with torch.no_grad():
            val_loss = self.mse_critirion(self.ridge(X_ts), y_ts).numpy()
        return {"score": val_loss}


# writer = SummaryWriter(log_dir="tb_logs")
# for x in range(200):
#     out = ridge(X_ts)
#     loss = mse_critirion(out, y_ts)
#     if x % 10 == 0:
#         writer.add_scalar("loss", loss, x)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# with torch.no_grad():
#     val_loss = mse_critirion(ridge(X_ts), y_ts)
#     print(val_loss)

# Create a HyperOpt search space
config = {f"weight_decay_{x}": tune.loguniform(1e-2, 0.5) for x in range(13)}
config.update({"lr": tune.loguniform(1e-3, 0.1)})

# Specify the search space and minimize score
# hyperopt = HyperOptSearch(metric="score", mode="min")
hyperopt = OptunaSearch(metric="score", mode="min")
asha = ASHAScheduler(metric="score", mode="min")
# Execute 10 trials using HyperOpt and stop after 10 iterations
analysis = tune.run(
    RegTrainable,
    config=config,
    search_alg=hyperopt,
    scheduler=asha,
    num_samples=10,
    stop={"training_iteration": 10},
    verbose=2,
)
result = analysis.results_df.sort_values("score", ascending=True)
print(result)
best_config = result.loc[:, result.columns.str.startswith("config.")].iloc[0]
best_config.index = best_config.index.str.lstrip("config.")

ridge_alpha = best_config[best_config.index.str.startswith("weight_decay_")]
ridge_lr = best_config["lr"]


def ridge_test_error(X_train, y_train, X_test, y_test, alpha, lr):
    """
    ridge regression test error
    """
    ridge = GroupRidge(13, 1)
    mse_critirion = nn.MSELoss()
    opt_param = [
        {"params": [ridge.group_weight[x]], "weight_decay": alpha[x]} for x in range(13)
    ]
    opt_param += [{"params": ridge.bias, "weight_decay": 0}]
    optimizer = SGD(opt_param, lr=lr)
    for _ in range(200):
        out = ridge(X_train)
        loss = mse_critirion(out, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return mse_critirion(ridge(X_test), y_test).numpy()


# def loo_error(alpha, lr):
#     """
#     leave one out cv error
#     """
#     loo = LeaveOneOut()
#     errors = np.array([])
#     for train_id, test_id in loo.split(Xs):
#         X_train, y_train = Xs[train_id], y[train_id]
#         X_test, y_test = Xs[test_id], y[test_id]
#         X_train = torch.as_tensor(X_train, dtype=torch.float)
#         X_test = torch.as_tensor(X_test, dtype=torch.float)
#         y_train = torch.as_tensor(y_train, dtype=torch.float).view(-1, 1)
#         y_test = torch.as_tensor(y_test, dtype=torch.float).view(-1, 1)
#         test_error = ridge_test_error(X_train, y_train, X_test, y_test, alpha, lr)
#         errors = np.append(errors, test_error)
#     return errors


# ridge_loo = loo_error(ridge_alpha, ridge_lr)
# 23.84181242809826


class PLGroupRidge(pl.LightningModule):
    """
    group ridge module
    """

    def __init__(self, in_features, out_features, group=None, bias=True):
        """
        init method
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group = group
        if group is None:
            self.group_weight = nn.ParameterList(
                [
                    nn.parameter.Parameter(torch.Tensor(out_features, 1))
                    for _ in range(in_features)
                ]
            )
        else:
            group_size = self._parse_group()
            self.group_weight = nn.ParameterList(
                [
                    nn.parameter.Parameter(torch.Tensor(out_features, x))
                    for x in group_size
                ]
            )
        self.weight = torch.cat(list(self.group_weight), dim=1)
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def _parse_group(self):
        """
        parse group argument
        """
        group_size = [len(x) for x in self.group]
        assert self.in_features == sum(
            group_size
        ), "Input dimension and group size mismatch!"
        elements = itertools.chain.from_iterable(self.group)
        assert all(
            0 <= x < self.in_features for x in elements
        ), "Elements of group MUST be less than or equal to in_features."
        return group_size

    def reset_parameters(self) -> None:
        for param in self.group_weight:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.weight = torch.cat(list(self.group_weight), dim=1)
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

    def configure_optimizers(self):
        """
        optimizers
        """
        opt_param = [
            {"params": [self.group_weight[x]], "weight_decay": 0.1} for x in range(13)
        ]
        opt_param += [{"params": self.bias, "weight_decay": 0}]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return F.mse_loss(y_hat, y)


# design matrix 1
# ob3: 1 2 3 4 5
# ob2: 2 3 4 5 6

# design matrix 2
# ob3: 23 24 23 24 25
# ob3: A1*a,B1*b A2*c,B2*d e,f ...
# ob3: a,b c,d e,f ...

# coeff 1
# ob3: A1,B1 A2,B2 ...

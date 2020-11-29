import itertools
import math

from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW, SGD
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

from boston_ts import X, y

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
X_ts = torch.as_tensor(Xs, dtype=torch.float)
y_ts = torch.as_tensor(y, dtype=torch.float).view(-1, 1)


class RidgeReg(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, group=None):
        """
        init method
        """
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        assert dim_in == sum(
            len(x) for x in group.items()
        ), "Input dimension and groups mismatch!"
        if group is None:
            self.weight = nn.parameter.Parameter(torch.Tensor(dim_out, dim_in))
        else:
            self.beta = nn.ParameterDict({nn.parameter.Parameter(torch.Tensor())})

    def _parse_group(self):
        """
        parse group argument
        """
        pass

    def forward(self, input):
        """
        forward method
        """
        return self.beta(input)


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


ridge = GroupRidge(13, 1)
mse_critirion = nn.MSELoss()
opt_param = [{"params": [ridge.group_weight[x]], "weigth_decay": 0.1} for x in range(13)]
optimizer = SGD(opt_param, lr=0.001)
# optimizer = SGD(ridge.parameters(), lr=0.01, weight_decay=0.1)
# optimizer = SGD(ridge.group_weight[0], lr=0.01, weight_decay=0.1)


class RegTrainable(tune.Trainable):
    def setup(self, config):
        # config (dict): A dict of hyperparameters
        self.ridge = GroupRidge(13, 1)
        self.mse_critirion = nn.MSELoss()
        self.optimizer = AdamW(
            self.ridge.parameters(), lr=config["lr"], weight_decay=config["l2_reg"]
        )

    def step(self):  # This is called iteratively.
        for _ in range(200):
            out = self.ridge(X_ts)
            loss = self.mse_critirion(out, y_ts)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        with torch.no_grad():
            val_loss = self.mse_critirion(self.ridge(X_ts), y_ts).numpy()
        return {"score": val_loss}


for _ in range(1000):
    out = ridge(X_ts)
    loss = mse_critirion(out, y_ts)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    val_loss = mse_critirion(ridge(X_ts), y_ts)
    print(val_loss)

# Create a HyperOpt search space
config = {"l2_reg": tune.loguniform(1e-2, 0.5), "lr": tune.loguniform(1e-3, 0.1)}

# Specify the search space and minimize score
hyperopt = HyperOptSearch(metric="score", mode="min")
asha = ASHAScheduler(metric="score", mode="min")
# Execute 20 trials using HyperOpt and stop after 20 iterations
analysis = tune.run(
    RegTrainable,
    config=config,
    search_alg=hyperopt,
    scheduler=asha,
    num_samples=100,
    stop={"training_iteration": 50},
)
print(analysis.results_df.sort_values("score", ascending=True))

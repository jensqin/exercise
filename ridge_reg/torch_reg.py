from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch import tensor
from torch.optim import AdamW, SGD
from ignite.engine import Engine, Events, create_supervised_trainer

from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

from boston_ts import X, y

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
X_ts = torch.as_tensor(Xs, dtype=torch.float)
y_ts = torch.as_tensor(y, dtype=torch.float).view(-1, 1)


class RidgeReg(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, grouper=0):
        """
        init method
        """
        super().__init__()
        self.beta = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        """
        forward method
        """
        return self.beta(x)


# ridge = nn.Linear(13, 1)
# mse_critirion = nn.MSELoss()
# optimizer = SGD(ridge.parameters(), lr=0.001, weight_decay=0.1)


class RegTrainable(tune.Trainable):
    def setup(self, config):
        # config (dict): A dict of hyperparameters
        self.ridge = RidgeReg(13, 1)
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


# for epoch in range(1000):
#     if epoch % 999 == 0:
#         print(f"epoch: {epoch}")
#     out = ridge(X_ts)
#     loss = mse_critirion(out, y_ts)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# with torch.no_grad():
#     val_loss = mse_critirion(ridge(X_ts), y_ts)
#     print(val_loss)

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

from sklearn.datasets import load_boston
from sklearn import metrics
import pandas as pd
import pyro
import torch
from torch import nn
from pyro.nn import PyroModule, PyroSample
from pyro import distributions as dist
from pyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive


class BayesianRegression(PyroModule):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_dim, out_dim)
        self.linear.weight = PyroSample(
            dist.Normal(0.0, 1.0).expand([out_dim, in_dim]).to_event(2)
        )
        self.linear.bias = PyroSample(
            dist.Normal(0.0, 10.0).expand([out_dim]).to_event(1)
        )

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 10.0))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean


model = BayesianRegression(13, 1)
guide = AutoDiagonalNormal(model)
# guide = AutoLowRankMultivariateNormal(model)

adam = pyro.optim.ClippedAdam({"lr": 0.03})
svi = SVI(model, guide, adam, loss=Trace_ELBO())

X, y = load_boston(return_X_y=True)
X_ts = torch.as_tensor(X, dtype=torch.float)
y_ts = torch.as_tensor(y, dtype=torch.float)
num_iterations = 2000
pyro.clear_param_store()
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(X_ts, y_ts)
    if j % 100 == 0:
        print(f"[iteration {j+1}] loss: {loss / len(y)}")


def summary_stats(samples):
    return {
        k: {
            "mean": torch.mean(v, 0).detach().numpy(),
            "std": torch.std(v, 0).detach().numpy(),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0].detach().numpy(),
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0].detach().numpy(),
        }
        for k, v in samples.items()
    }


predictive = Predictive(
    model,
    guide=guide,
    num_samples=800,
    return_sites=("linear.weight", "obs", "_RETURN"),
)

samples = predictive(X_ts)
pred_summary = summary_stats(samples)
mu = pred_summary["_RETURN"]
yhat = pred_summary["obs"]
predictions = pd.DataFrame(
    {
        "mu_mean": mu["mean"],
        "mu_perc_5": mu["5%"],
        "mu_perc_95": mu["95%"],
        "y_mean": yhat["mean"],
        "y_perc_5": yhat["5%"],
        "y_perc_95": yhat["95%"],
        "true_y": y,
    }
)

metrics.mean_squared_error(mu["mean"], y)

import pandas as pd
import torch
import pyro
from pyro import nn
from pyro import distributions as dist
from pyro.infer import autoguide, SVI, Trace_ELBO
from sklearn.datasets import load_boston

boston = load_boston()


def sklearn_df(data):
    x = pd.DataFrame(data["data"], columns=data["feature_names"])
    y = data["target"]
    return x, y


boston_x, boston_y = sklearn_df(boston)


class BostonBayes(pyro.nn.PyroModule):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.PyroModule[torch.nn.Linear](in_dim, out_dim)
        self.linear.weight = nn.PyroSample(
            dist.Normal(0.0, 1.0).expand([out_dim, in_dim]).to_event(2)
        )
        self.linear.bias = nn.PyroSample(
            dist.Normal(0.0, 10.0).expand([out_dim]).to_event(1)
        )

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 10.0))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean


model = BostonBayes(len(boston_x.columns), 1)
guide = autoguide.AutoLowRankMultivariateNormal(model)
optimizer = pyro.optim.ClippedAdam({"lr": 0.03})
svi = SVI(model=model, guide=guide, optim=optimizer, loss=Trace_ELBO())

pyro.clear_param_store()
num_iterations = 50
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(
        torch.as_tensor(boston["data"], dtype=torch.float),
        torch.as_tensor(boston["target"], dtype=torch.float),
    )
    if j % 10 == 0:
        print(f"[iteration {j+1}] loss: {loss/len(boston['target'])}")

guide.requires_grad_(False)

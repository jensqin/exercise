from pyro.infer.autoguide.guides import AutoDiagonalNormal
import torch
import pyro
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from pyro.optim import AdamW
from pyro.nn import PyroModule, PyroSample
from pyro import distributions as dist
from pyro.infer import autoguide, SVI, Trace_ELBO
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from ignite.handlers import TerminateOnNan
from ignite.engine import Engine, Events, create_supervised_trainer

X, y = load_boston(return_X_y=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)


class BostonBayes(pyro.nn.PyroModule):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_dim, out_dim)
        self.linear.weight = PyroSample(
            dist.Normal(0.0, 3.0).expand([out_dim, in_dim]).to_event(2)
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


class BostonDataModule(pl.LightningDataModule):
    """
    boston data
    """

    def __init__(self, X, y):
        super().__init__()
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def train_dataloader(self):
        """
        train dataloader
        """
        ds = TensorDataset(self.X, self.y)
        return DataLoader(ds, batch_size=506)

    def val_dataloader(self):
        """
        val dataloader
        """
        ds = TensorDataset(self.X, self.y)
        return DataLoader(ds, batch_size=506)


def create_trainer(svi):
    """create ignite trainer"""

    # update func
    def update_model(engine, batch):
        model.train()
        x, y = batch
        loss = svi.step(x, y)
        return loss / len(y)

    trainer = Engine(update_model)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    return trainer


pyro.clear_param_store()
model = BostonBayes(len(X[0]), 1)
bos = BostonDataModule(X, y)
guide = autoguide.AutoLowRankMultivariateNormal(model)
optimizer = AdamW({"lr": 3e-2})
svi = SVI(model=model, guide=guide, optim=optimizer, loss=Trace_ELBO())

trainer = create_trainer(svi)


@trainer.on(Events.EPOCH_COMPLETED)
def print_loss(engine):
    """
    docstring
    """
    print(engine.state.output)


loader = bos.train_dataloader()
trainer.run(loader, max_epochs=50)


num_iterations = 50
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    if j % 1 == 0:
        print(f"[iteration {j+1}] loss: {loss/len(y)}")


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
import pytorch_lightning as pl

X, y = load_boston(return_X_y=True)


class BostonBayes(pyro.nn.PyroModule):
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


model = BostonBayes(len(X[0]), 1)
guide = autoguide.AutoLowRankMultivariateNormal(model)
optimizer = AdamW({"lr": 0.03})
svi = SVI(model=model, guide=guide, optim=optimizer, loss=Trace_ELBO())

pyro.clear_param_store()
num_iterations = 50
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    if j % 10 == 0:
        print(f"[iteration {j+1}] loss: {loss/len(y)}")

# guide.requires_grad_(False)


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
        return DataLoader(ds, batch_size=32)

    def val_dataloader(self):
        """
        val dataloader
        """
        ds = TensorDataset(self.X, self.y)
        return DataLoader(ds, batch_size=200)


# class PyroOptWrap(pyro.infer.SVI):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def state_dict(self,):
#         return {}


class PyroLightning(pl.LightningModule):
    """
    pyro lightning module
    """

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

    def model(self, batch):
        """
        model
        """
        x, y = batch
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 10.0))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean

    def training_step(self, batch, batch_idx):
        """
        training step
        """
        x, y = batch
        return self.loss_fn(self.model, self.guide, x, y)

    def configure_optimizers(self):
        """
        configure optimizers
        """
        self.loss_fn = Trace_ELBO().differentiable_loss
        self.guide = autoguide.AutoDiagonalNormal(self.model)
        return torch.optim.AdamW(self.linear.named_pyro_params(), lr=0.03)


bos = BostonDataModule(X, y)
model = PyroLightning(13, 1)
trainer = pl.Trainer(min_epochs=1, max_epochs=5)
pyro.clear_param_store()
trainer.fit(model, datamodule=bos)

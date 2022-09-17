import torch
from torch import distributions as dist

# import pyro
# from pyro import distributions as dist

norm = torch.distributions.Normal(0, 1)

# rsample refers to reparameterized sample
x = norm.rsample((2, 2))
y = norm.sample((2, 2))

# batch shape and event shape
# independent
loc = torch.zeros(3)
scale = torch.ones(3)
mvn = dist.MultivariateNormal(loc, torch.diag(scale))
mvn.batch_shape
mvn.event_shape
norm = dist.Normal(loc, scale)
norm.batch_shape
norm.event_shape
diagn = dist.Independent(norm, reinterpreted_batch_ndims=1)
diagn.batch_shape
diagn.event_shape

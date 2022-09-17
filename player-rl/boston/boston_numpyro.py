from jax import random
from jax import numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.infer import NUTS, MCMC

# from numpyro.infer.reparam import TransformReparam

from boston_ts import X, y


def regression_hs(
    X, scale_icept, scale_noise, scale_global, nu_global, nu_local, obs=None
):
    """
    sparse regression with horseshole prior
    """
    with numpyro.plate("coefficients", X.shape[1]):
        z_vec = numpyro.sample("z_vec", dist.Normal(0.0, 1.0))
        r1_local = numpyro.sample("r1_local", dist.Normal(0.0, 1.0))
        r2_local = numpyro.sample(
            "r2_local", dist.InverseGamma(0.5 * nu_local, 0.5 * nu_local)
        )
    sigma = numpyro.sample("sigma", dist.HalfCauchy(0.0, scale_noise))
    r1_global = numpyro.sample("r1_global", dist.Normal(0.0, scale_global * sigma))
    r2_global = numpyro.sample(
        "r2_global", dist.InverseGamma(0.5 * nu_global, 0.5 * nu_global)
    )
    lambda_local = r1_local * jnp.sqrt(r2_local)
    tau = r1_global * jnp.sqrt(r2_global)
    beta = z_vec * lambda_local * tau
    beta0 = numpyro.sample("beta0", dist.Normal(0.0, scale_icept))
    mu = jnp.matmul(beta, X) + beta0
    with numpyro.plate("data", len(X)):
        return numpyro.sample("obs", dist.Normal(mu, sigma), obs=obs)


def default_scale_global(n, D, p):
    """
    default global scale of horseshoe prior
    """
    return p * max(D - p, 1) / jnp.sqrt(n)


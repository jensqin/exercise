import time
import pickle
from argparse import ArgumentParser
from typing import Union

import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import numpy as jnp
from jax import random
from numpy.lib.npyio import load
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, Predictive
from sklearn import metrics

from utils import load_nba, transform_to_array

NDArrayType = Union[np.ndarray, jnp.ndarray]


# def scalar_embedding(embedding: NDArrayType, index: int) -> float:
#     """
#     get random effect
#     """
#     return embedding[index]


def vectorized_embedding(embedding: NDArrayType, index: NDArrayType) -> np.ndarray:
    """
    vectorized random effect
    """
    return jnp.vectorize(lambda emb, id: emb[id], excluded=(0,))(embedding, index)


def nba_model(
    x: NDArrayType,
    y: NDArrayType = None,
    n_team: int = 5,
    n_player: int = 85,
    n_group: int = 7,
    scale_global: float = 1,
    # nu_global: float = 1,
    # nu_local: float = 1,
) -> jnp.ndarray:
    """
    nba numpyro model
    """
    # n_team_emb, n_player_emb = 1, 1
    scale_icept = 10
    local_vec = jnp.ones(n_group)
    tau = numpyro.sample("tau", dist.HalfCauchy(scale_global))
    lam = numpyro.sample("lambda", dist.HalfCauchy(local_vec).to_event())
    sig = lam * tau
    sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
    bias = numpyro.sample("bias", dist.StudentT(4, 0.0, 1.0))
    b_fc = numpyro.sample("fc", dist.Normal(0.0, 1.0).expand([2]).to_event(1))
    emb_offt = numpyro.sample("offt", dist.Normal(0.0, 1.0).expand([n_team]))
    emb_deft = numpyro.sample("deft", dist.Normal(0.0, 1.0).expand([n_team]))
    emb_offp = numpyro.sample("offp", dist.Normal(0.0, 1.0).expand([n_player]))
    emb_defp = numpyro.sample("defp", dist.Normal(0.0, 1.0).expand([n_player]))
    emb_offpa = numpyro.sample("offpa", dist.Normal(0.0, 1.0).expand([n_player]))
    emb_defpa = numpyro.sample("defpa", dist.Normal(0.0, 1.0).expand([n_player]))
    fc, offt, deft, offp, offpa, defp, defpa = x
    fc_ = jnp.dot(fc, b_fc.T)
    offt = offt.ravel()
    deft = deft.ravel()
    offt_ = vectorized_embedding(emb_offt, offt)
    deft_ = vectorized_embedding(emb_deft, deft)
    offp_ = vectorized_embedding(emb_offp, offp)
    defp_ = vectorized_embedding(emb_defp, defp)
    offpa_ = vectorized_embedding(emb_offpa, offp) * offpa
    defpa_ = vectorized_embedding(emb_defpa, defp) * defpa
    offp_ = jnp.sum(offp_, axis=1)
    defp_ = jnp.sum(defp_, axis=1)
    offpa_ = jnp.sum(offpa_, axis=1)
    defpa_ = jnp.sum(defpa_, axis=1)
    mu = (
        bias * scale_icept
        + fc_ * sig[0]
        + offt_ * sig[1]
        + deft_ * sig[2]
        + offp_ * sig[3]
        + defp_ * sig[4]
        + offpa_ * sig[5]
        + defpa_ * sig[6]
    )
    mu = mu.ravel()
    with numpyro.plate("data", fc.shape[0]):
        _ = numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
    return mu


def run_inference(
    x: NDArrayType,
    y: NDArrayType,
    n_warmup: int = 30,
    n_samples: int = 30,
    n_chains: int = 1,
    rng_key=random.PRNGKey(0),
) -> dict:
    """
    run inference
    """
    kernel = NUTS(nba_model, target_accept_prob=0.8, max_tree_depth=10)
    mcmc = MCMC(kernel, n_warmup, n_samples, n_chains)
    mcmc.run(rng_key, x=x, y=y)
    mcmc.print_summary()
    return mcmc.get_samples()


def posterior_predict(samples: dict, X: NDArrayType, rng_key=random.PRNGKey(0)) -> dict:
    """
    get predictions from samples
    """
    pred = Predictive(nba_model, samples)
    return pred(rng_key, X)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_warmup", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_chains", type=int, default=1)
    args = parser.parse_args()
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    # train, test = load_nba(split_mode="test", test=0.2)
    train = load_nba(path="data/nba_train.csv")
    test = load_nba(path="data/nba_test.csv")
    X_train, y_train = transform_to_array(train, to_tensor=False)
    X_test, y_test = transform_to_array(test, to_tensor=False)
    # samples = run_inference(
    #     X_train, y_train, args.n_warmup, args.n_samples, args.n_chains, rng_key=rng_key
    # )
    start = time.time()
    samples = run_inference(X_train, y_train, rng_key=rng_key)
    print(f"Runtime: {time.time() - start}")
    predictions = posterior_predict(samples, X_test, rng_key=rng_key_predict)
    preds = jnp.mean(predictions["obs"], axis=0)
    print(metrics.mean_squared_error(preds, y_test))

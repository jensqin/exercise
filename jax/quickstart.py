import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random


key = random.PRNGKey(0)
x = random.normal(key, (10,))


@jit
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))


x_small = jnp.arange(3.0)
derivative_fn = grad(sum_logistic)
derivative_fn(x_small)

x_shape, n_group = (8, 2), 5
group = random.randint(key, x_shape, minval=0, maxval=n_group)
emb = random.normal(key, (n_group,))
x_group = jnp.ones(x_shape)


def scalar_random_effect(emb, group, x_group):
    """
    get random effect
    """
    return x_group * emb[group]


def row_random_effect(emb, group, x_group):
    """
    get random effect
    """
    return x_group * jnp.array([emb[i] for i in group])


def vmap_random_effect(emb, group, x_group):
    """
    vmap random effect
    """
    return vmap(row_random_effect, in_axes=(None, 1, 1), out_axes=1)(
        emb, group, x_group
    )


def vec_random_effect(emb, group, x_group):
    """
    vectorized random effect
    """
    return jnp.vectorize(scalar_random_effect, excluded=(0,))(emb, group, x_group)

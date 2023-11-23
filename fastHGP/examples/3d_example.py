import jax.numpy as jnp
import jax.random as jr
import gpjax as gpx
from fastHGP.kernels import LaplaceBF

def generate_data(kernel, N=300, key=jr.PRNGKey(13)):
    x3d = jr.uniform(key, shape=(300, 3))*2-1
    meanf = gpx.mean_functions.Zero()
    kernel = kernel.replace(lengthscale=jnp.ones((3,)) * kernel.lengthscale)
    fprior = gpx.Prior(mean_function=meanf, kernel=kernel)
    f = fprior.sample_approx(num_samples=1, key=key)
    y3d = f(x3d)
    return gpx.Dataset(X=x3d, y=y3d)

def build_bf(m, L, center):
    ms = jnp.ones((3,)) * m
    Ls = jnp.ones((3,)) * L
    center = jnp.ones((3,)) * center
    return LaplaceBF(ms, Ls, center)
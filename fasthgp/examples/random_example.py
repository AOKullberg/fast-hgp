# Random 3D example (drawn from a prior)

import jax.numpy as jnp
import jax.random as jr
import gpjax as gpx
from fasthgp.kernels import LaplaceBF

def generate_data(kernel, N=300, key=jr.PRNGKey(13)):
    x3d = jr.uniform(key, shape=(N, 3))*2-1
    meanf = gpx.mean_functions.Zero()
    kernel = kernel.replace(lengthscale=jnp.ones((3,)) * kernel.lengthscale)
    fprior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    f = fprior.sample_approx(num_samples=1, key=key)
    y3d = f(x3d)
    xtest = jnp.vstack([x.flatten() for x in jnp.meshgrid(*([jnp.linspace(-1, 1, 15)] * 3))]).T
    ytest = f(xtest)
    return gpx.Dataset(X=x3d, y=y3d), gpx.Dataset(X=xtest, y=ytest)

def build_bf(m, L, center):
    ms = jnp.ones((3,)) * m
    Ls = jnp.ones((3,)) * L
    center = jnp.ones((3,)) * center
    return LaplaceBF(ms, Ls, center)
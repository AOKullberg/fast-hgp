# Precipitation data set

import jax.numpy as jnp
import jax.random as jr
import scipy.io as sio
import gpjax as gpx
from fasthgp.kernels import LaplaceBF
import os

def generate_data():
    data = sio.loadmat(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/data.mat"))
    X, y = data['x'], data['y']
    boundary = jnp.array([X.min(axis=0), X.max(axis=0)])
    return gpx.Dataset(X - boundary.mean(axis=0), y) # Center the data for the HGP, y is already centered

def build_bf(m, D):
    boundary = jnp.array([D.X.min(axis=0), D.X.max(axis=0)])
    hgp_boundary = boundary + jnp.array([[-1], [1]]) * jnp.diff(boundary, axis=0)*0.1
    Ls = jnp.diff(hgp_boundary, axis=0) / 2
    ms = jnp.ones((2,)) * m
    return LaplaceBF(num_bfs=ms, L=Ls)
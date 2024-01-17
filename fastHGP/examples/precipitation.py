import jax.numpy as jnp
import jax.random as jr
import scipy.io as sio
import gpjax as gpx
from fastHGP.kernels import LaplaceBF
import os

def generate_data():
    data = sio.loadmat(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/data.mat"))
    return gpx.Dataset(data['x'], data['y'])

def build_bf(m, D):
    boundary = jnp.array([D.X.min(axis=0), D.X.max(axis=0)])
    hgp_centers = boundary.mean(axis=0)
    hgp_boundary = boundary + jnp.array([[-1], [1]]) * jnp.diff(boundary, axis=0)*0.1
    Ls = jnp.diff(hgp_boundary, axis=0) / 2
    ms = jnp.ones((2,)) * m
    return LaplaceBF(ms, Ls, hgp_centers)
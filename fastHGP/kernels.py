from dataclasses import dataclass
from itertools import product
import jax
import jax.numpy as jnp
from gpjax.base import param_field
import gpjax as gpx
from jaxtyping import Float, Array

@dataclass
class LaplaceBF(gpx.base.Module):
    num_bfs: Float[Array, "M"] = param_field(default=jnp.ones((1,)), trainable=False)
    L: Float[Array, "M"] = param_field(default=jnp.ones((1,)), trainable=False)
    center: Float[Array, "M"] = param_field(default=jnp.zeros((1,)), trainable=False)

    def __post_init__(self):
        try:
            self.j = [jnp.arange(1, int(m) + 1) for m in self.num_bfs]
        except:
            self.j = jnp.arange(1, self.num_bfs + 1)
        self.js = jnp.array(list(product(*[x.flatten() for x in self.j])))
        self.num_bfs = jnp.atleast_1d(jnp.array(self.num_bfs)).astype(int)
        D  = len(self.num_bfs)
        self.L = jnp.atleast_1d(jnp.array(self.L))
        self.center = jnp.atleast_1d(jnp.array(self.center))
        if len(self.L) != D:
            self.L = jnp.ones((D,)) * self.L[0]
        if len(self.center) != D:
            self.center = jnp.ones((D,)) * self.center[0]
    
    @property
    def M(self):
        return jnp.prod(self.num_bfs).astype(int)
        
    def _1d_call(self, x, j):
        z = x - self.center
        L = self.L
        op = jnp.pi * j * (z + L) / (2 * L)
        return jnp.prod(1 / jnp.sqrt(L) * jnp.sin(op) * (z >= -L) * (z <= L))

    def __call__(self, x):
        return jax.vmap(jax.vmap(self._1d_call, (None, 0), 0), (0, None), 0)(x, self.js)

    def eigenvalues(self):
        j = self.js
        L = self.L
        lambda_j = ( jnp.pi * j / (2 * L) )**2
        return lambda_j


@dataclass
class SE(gpx.kernels.RBF):
    def spectral_density(self, x):
        """The spectral density of a squared exponential kernel.

        Parameters
        ----------
        lengthscale : ScalarFloat
            The lengthscale of the kernel.
        variance : ScalarFloat
            The variance of the kernel
        x : Float[Array, "N 1"]
            Input data

        Returns
        -------
        Float[Array, "N 1"]
            The spectral density of the kernel at specific inputs.
        """    
        l, s2f = self.lengthscale, self.variance
        return jnp.prod(s2f * jnp.sqrt( 2 * jnp.pi * l**2) * jnp.exp( - 1 / 2 * x**2 * l**2))
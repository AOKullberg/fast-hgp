from abc import abstractmethod
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from gpjax.base import param_field, Module
from jaxtyping import Float, Array
from .utils import integrate

@dataclass
class AbstractCost(Module):
    @abstractmethod
    def __call__(self, test_input, gp):
        raise NotImplementedError

@dataclass
class DualCost(AbstractCost):
    def __call__(self, test_input, gp):
        alpha = gp.alpha
        B = gp.B_diag
        lambda_j = gp.bf.eigenvalues()
        Lambdainv = 1/jax.vmap(gp.prior.kernel.spectral_density, 0, 0)(jnp.sqrt(lambda_j))
        # Lambdainv = 1/jnp.prod(jnp.atleast_2d(self.prior.kernel.spectral_density(jnp.sqrt(lambda_j))), axis=0)
        mi = ((1/(B + Lambdainv)) * alpha)**2 # Approximate mean of each component
        inds = jnp.flipud(jnp.argsort(mi))
        cost = mi[inds] # The delta cost of leaving out each component
        return cost, inds # Sorted cost of each component with inds indicating the sorting

@dataclass
class IntegralCost(AbstractCost):
    limits: Float[Array, "2 M"] = param_field(default=[0, 1], trainable=False)
    N: Float = param_field(default=100, trainable=False)

    def __call__(self, test_input, gp):
        test_mu = test_input.mean(axis=0) # Surrogate test point
        a, b = test_mu + self.limits
        R = integrate(lambda x: gp.bf(x)**2, jnp.vstack([a, b]))
        alpha = gp.alpha
        B = gp.B_diag
        lambda_j = gp.bf.eigenvalues()
        Lambdainv = 1/jax.vmap(gp.prior.kernel.spectral_density, 0, 0)(jnp.sqrt(lambda_j))
        # Lambdainv = 1/jnp.prod(jnp.atleast_2d(self.prior.kernel.spectral_density(jnp.sqrt(lambda_j))), axis=0)
        mi = ((1/(B + Lambdainv)) * alpha)**2 # Approximate mean of each component
        cost = mi * R
        inds = jnp.flipud(jnp.argsort(cost))
        return cost[inds], inds
        

@dataclass
class Selector(Module):
    cost: AbstractCost
    def __call__(self, test_input, gp):
        pass

@dataclass
class DifferenceSelector(Selector):
    tolerance: float = param_field(default=1e-3, trainable=False)
    def __call__(self, test_input, gp):
        cost, inds = self.cost(test_input, gp)
        Nj = jnp.argmax(cost < self.tolerance)
        return inds[:Nj]

@dataclass
class BoundSelector(Selector):
    tolerance: float = param_field(default=1., trainable=False)
    def __call__(self, test_input, gp):
        cost, inds = self.cost(test_input, gp)
        C = jnp.cumsum(cost)
        Nj = jax.lax.cond(C[-1] < self.tolerance, 
                     lambda: inds.shape[0], 
                     lambda: jnp.argmax(C > self.tolerance))
        return inds[:Nj]

@dataclass
class FractionSelector(Selector):
    fraction: float = param_field(default=1e-1, trainable=False)
    def __call__(self, test_input, gp):
        _, inds = self.cost(test_input, gp)
        N = inds.shape[0]
        Nj = jnp.ceil(jnp.array(N * self.fraction)).astype(int)
        return inds[:Nj]

@dataclass
class FixedSelector(Selector):
    Nj: int = param_field(default=10, trainable=False)
    def __call__(self, test_input, gp):
        _, inds = self.cost(test_input, gp)
        N = inds.shape[0]
        Nj = min([N, self.Nj])
        return inds[:Nj]
    
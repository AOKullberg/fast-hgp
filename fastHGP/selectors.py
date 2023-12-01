from dataclasses import dataclass
import jax
import jax.numpy as jnp
from gpjax.base import param_field

@dataclass
class Selector:
    def __call__(self, cost, sorted_inds):
        pass

@dataclass
class DifferenceSelector(Selector):
    tolerance: float = param_field(default=1e-3, trainable=False)
    def __call__(self, cost, sorted_inds):
        Nj = jnp.argmax(cost < self.tolerance)
        return sorted_inds[:Nj]

@dataclass
class BoundSelector(Selector):
    tolerance: float = param_field(default=1, trainable=False)
    def __call__(self, cost, sorted_inds):
        C = jnp.cumsum(cost)
        Nj = jax.lax.cond(C[-1] < self.tolerance, 
                     lambda: sorted_inds.shape[0], 
                     lambda: jnp.argmax(C > self.tolerance))
        return sorted_inds[:Nj]

@dataclass
class FractionSelector(Selector):
    fraction: float = param_field(default=1e-1, trainable=False)
    def __call__(self, cost, sorted_inds):
        N = sorted_inds.shape[0]
        Nj = jnp.ceil(jnp.array(N * self.fraction)).astype(int)
        return sorted_inds[:Nj]

@dataclass
class FixedSelector(Selector):
    Nj: int = param_field(default=10, trainable=False)
    def __call__(self, cost, sorted_inds):
        N = sorted_inds.shape[0]
        Nj = min([N, self.Nj])
        return sorted_inds[:Nj]
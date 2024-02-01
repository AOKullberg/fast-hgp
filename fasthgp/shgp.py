from dataclasses import dataclass
from itertools import product
import jax.numpy as jnp
import gpjax as gpx
from gpjax.base import param_field, static_field
from gpjax.typing import ScalarFloat
from jaxtyping import Float, Array
from .utils import (
    TB, 
    TB_diag, 
    B, 
    B_diag, 
    gamma
)
from .kernels import LaplaceBF
from .gp_utils import (
    dual_to_mean
)
from .hgp import HGP
from fasthgp.selectors import (
    Selector,
    BoundSelector,
    DualCost
)

@dataclass 
class SHGP(gpx.gps.AbstractPosterior):
    """An N-D implementation of a "sparse" HGP, i.e., a HGP where only the necessary statistics are stored.
    
    The SHGP stores (alpha, Gamma) where alpha is the dual to the mean of an HGP and Gamma is an array of sufficient statistics in order to create B, which is the dual of the covariance of an HGP.

    Further, the SHGP attains fast predictions by approximately selecting the most important basis functions according to the criterion
    $$
    L(\theta) = | f(t) - \hat{f}(t) |_2^2
    $$
    where $\theta$ corresponds to the parameters of the posterior distribution of the GP (m, S). Assuming that the components we select are contained in $J$, the criterion can be upper bounded as 
    $$
    L \leq \sum_{j\notin J} | m_j |^2,
    $$
    and we can thus select the components corresponding to the largest mean values.
    The number of components are automatically selected such that components with a mean of atleast the tolerance are included.

    Parameters
    ----------
    bf : LaplaceBF
        The basis functions to use.
    jitter : float
        The jitter to use when inverting possibly singular matrices
    alpha : 
        First-order dual parameter
    Gamma : 
        Condensed second-order dual parameter (B in HGP)
    approximate_selector : fastHGP.Selector
        A basis function selector for approximate predictions -- not necessary.

    """
    bf: LaplaceBF = param_field(default=None, trainable=False)
    jitter: ScalarFloat = static_field(1e-6)
    alpha: Float[Array, "M"] = param_field(jnp.zeros((1,)), trainable=False)
    Gamma: Float[Array, "M**D"] = param_field(jnp.zeros((1,)), trainable=False)
    approximate_selector: Selector = param_field(default_factory=lambda: BoundSelector(DualCost()),
                                                 trainable=False)

    def __post_init__(self):
        """Initializes the (condensed) dual parameters.
        """        
        m = self.bf.num_bfs
        M = self.bf.M
        self.alpha = jnp.zeros((M,))
        self.unique_k = jnp.vstack([x.flatten() for x in jnp.meshgrid(*[jnp.arange(1-mi, 2*mi+1) for mi in m], indexing='ij')]).T.astype(jnp.float64)
        self.Gamma = jnp.zeros((self.unique_k.shape[0],))
        self.indices = self.bf.js

    def reduce(self, inds):
        """Reduce basis to the indices given by inds.

        Parameters
        ----------
        inds : array_like
            Indices of basis to reduce to.

        Returns
        -------
        SHGP
            An SHGP with reduced basis
        """        
        return self.replace(alpha=self.alpha[inds],
                            indices=self.indices[inds],
                            bf=self.bf.replace(js=self.bf.js[inds]))

    predict = HGP.predict

    @property
    def ep(self):
        # Creates all permutations necessary for gamma computation
        D = self.unique_k.shape[1]
        return jnp.vstack([x.flatten() for x in jnp.meshgrid(*jnp.array([[-1., 1.]]*D), indexing='ij')]).T

    @property
    def M(self):
        """Number of basis functions.

        Returns
        -------
        int
        """        
        return self.bf.M

    def update_with_batch(self, data):
        """ Update the posterior with batch of data.

        Parameters
        ----------
        data : gpx.Dataset
            Dataset to update with.

        Returns
        -------
        SHGP
            An updated SHGP with new posterior parameters
        """        
        g = gamma(data.X, self.unique_k, self.bf.L)
        Phi = self.bf(data.X)
        alpha = jnp.matmul(Phi.T, data.y).squeeze()
        return self.replace(Gamma=self.Gamma + g.reshape(self.Gamma.shape), 
                            alpha=self.alpha + alpha)

    @property
    def B(self):
        """Reconstructs the second-order dual parameter (precision matrix) from Gamma

        Returns
        -------
        Array
            B
        """        
        return B(self.Gamma, self.indices, self.bf.num_bfs, self.ep)

    @property
    def B_diag(self):
        """Diagonal part of the dual (precision matrix)

        Returns
        -------
        Array
            Diagonal part of B
        """        
        return B_diag(self.Gamma, self.indices, self.bf.num_bfs, self.ep)

    @property
    def dual_parameters(self):
        """Dual parameters of the HGP

        Returns
        -------
        (Array, Array)
            Dual parametrization of the HGP (alpha, B)
        """      
        return (self.alpha, self.B)

    @property
    def mean_parameters(self):
        """Mean parameters of the HGP

        Returns
        -------
        (Array, Array)
            Mean parametrization of the HGP (m, S)
        """      
        return dual_to_mean(self.alpha, self.B, self.bf, self.prior.kernel.spectral_density, self.likelihood.obs_stddev)

@dataclass
class TSHGP(SHGP):
    """
    Saves the "necessary statistics" in tensor format instead -- allows easier indexing and marginally faster computation. However, does not allow a basis reduction.
    """
    def __post_init__(self):
        m = self.bf.num_bfs
        D = len(m)
        self.alpha = jnp.zeros((jnp.prod(m).astype(int),))
        self.p = jnp.array(list(product(*[[-1, 1]]*D))) # permutations
        self.unique_k = jnp.vstack([x.flatten() for x in jnp.meshgrid(*[jnp.arange(1-mi, 2*mi+1) for mi in m])]).T
        d = int(jnp.ceil(self.unique_k.shape[0]**(1/D)))
        self.Gamma = jnp.zeros([d]*D)
        self.indices = self.bf.js
        # self.indices = jnp.vstack([x.flatten() for x in jnp.meshgrid(*self.bf.j)]).T

    @property
    def B(self):
        return TB(self.Gamma, self.indices, self.bf.num_bfs, self.p)

    @property
    def B_diag(self):
        return TB_diag(self.Gamma, self.indices, self.bf.num_bfs, self.p)

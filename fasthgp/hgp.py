from dataclasses import dataclass
import jax
import jax.numpy as jnp
import gpjax as gpx
from gpjax.distributions import GaussianDistribution
from gpjax.base import (
    static_field, 
    param_field
)
from gpjax.typing import ScalarFloat
from jaxtyping import (
    Float, 
    Array
)
import cola
from .kernels import LaplaceBF
from .project import (
    project, 
    project_regularized, 
    project_mm, 
    project_regularized_mm
)
from .gp_utils import (
    predict,
    vpredict,
    compute_dual,
    mean_to_dual,
    dual_to_mean
)

from fasthgp.selectors import (
    Selector,
    BoundSelector,
    DualCost
)


@dataclass
class HGP(gpx.gps.AbstractPosterior):
    bf: LaplaceBF = param_field(default=None, trainable=False)
    jitter: ScalarFloat = static_field(1e-6)
    alpha: Float[Array, "M"] = param_field(jnp.zeros((1,)), trainable=False)
    B: Float[Array, "M M"] = param_field(jnp.identity(1), trainable=False)
    approximate_selector: Selector = static_field(default_factory=lambda: BoundSelector(DualCost()))

    def __post_init__(self):
        """Initializes the dual parameters.
        """        
        M = self.bf.M
        self.alpha = jnp.zeros((M,))
        self.B = jnp.zeros((M, M))
    
    @property
    def M(self):
        """Number of basis functions.

        Returns
        -------
        int
        """        
        return self.bf.M
    
    def reduce(self, inds):
        """Reduces the basis to the given indices.

        Parameters
        ----------
        inds : array_like
            Indices of the basis functions to reduce the basis to.

        Returns
        -------
        HGP
            An HGP with a reduced basis.
        """        
        return self.replace(alpha=self.alpha[inds],
                            B = self.B[inds[None, :], inds[:, None]],
                            bf=self.bf.replace(js=self.bf.js[inds]))
    
    def predict(self, test_inputs, full_cov=True, approx=False):
        """Prediction at test inputs.

        Parameters
        ----------
        test_inputs : array_like
            Inputs at which to predict.
        full_cov : bool, optional
            Compute full covariance matrix or not (otherwise diagonal), by default True
        approx : bool, optional
            Approximate the predictions, by default False

        Returns
        -------
        (gpjax.distributions.GaussianDistribution, gpjax.distributions.GaussianDistribution)
            Returns both predictions at the current test inputs **as well** as the posterior **mean** parameters (for other purposes)
        """        
        if approx:
            inds = self.approximate_selector(test_inputs, self)
            gp = self.reduce(inds)
            m, S = gp.mean_parameters
            bf = gp.bf
        else:
            m, S = self.mean_parameters
            bf = self.bf
        if full_cov:
            mu, V = predict(m, S, bf, test_inputs)
            return GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()),
                                        scale=cola.ops.Dense(V)), \
                    GaussianDistribution(loc=jnp.atleast_1d(m.squeeze()),
                                         scale=cola.ops.Dense(S))
        else:
            mu, V = vpredict(m, S, bf, test_inputs)
            return GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()),
                                        scale=cola.ops.Dense(jnp.diag(V.squeeze()))), \
                    GaussianDistribution(loc=jnp.atleast_1d(m.squeeze()),
                                         scale=cola.ops.Dense(S))
    
    @property
    def B_diag(self):
        """Diagonal part of the dual (precision matrix)

        Returns
        -------
        Array
            Diagonal part of B
        """        
        return self.B.diagonal()

    def compute_dual(self, data):
        """Computes the dual parameters of the given data.

        Parameters
        ----------
        data : gpx.Dataset
            Dataset to compute dual parameters for.

        Returns
        -------
        (Array, Array)
            (alpha, B) of the given data set
        """        
        return compute_dual(self.bf, data)
        
    def update_with_batch(self, data):
        """ Update the posterior with batch of data.

        Parameters
        ----------
        data : gpx.Dataset
            Dataset to update with.

        Returns
        -------
        HGP
            An updated HGP with new posterior parameters
        """        
        alpha, B = self.compute_dual(data)
        return self.replace(
            alpha = alpha + self.alpha,
            B = B + self.B
            )

    def change_basis(self, new_bf, **kwargs):
        """Changes the basis of the HGP through projection.

        Parameters
        ----------
        new_bf : fasthgp.kernels.LaplaceBF
            Basis to change to.

        Returns
        -------
        HGP
            An HGP with a new basis 
        """        
        l = kwargs.get("l", self.prior.kernel.lengthscale)
        kwargs['limits'] = kwargs.get("limits", [-self.bf.L + l,
                                                 self.bf.L - l])
        m, S = project(self.bf, new_bf, self.mean_parameters, **kwargs)
        alpha, B = mean_to_dual(m, 
                                S, 
                                new_bf, 
                                self.prior.kernel.spectral_density, 
                                self.likelihood.obs_stddev)
        return self.replace(alpha=alpha, B=B, bf=new_bf)
    
    def change_basis_regularized(self, new_bf, **kwargs):
        """Regularized basis change of the HGP through projection.

        Regularizes the projection toward the prior (if you want to project to e.g. a new domain).

        Parameters
        ----------
        new_bf : fasthgp.kernels.LaplaceBF
            Basis to change to.

        Returns
        -------
        HGP
           An HGP with a new basis.
        """        
        l = kwargs.get("l", self.prior.kernel.lengthscale)
        kwargs['limits'] = kwargs.get("limits", [-self.bf.L + l,
                                                 self.bf.L - l])
        kwargs['prior_limits'] = kwargs.get("prior_limits", [[self.bf.L + l,
                                                            new_bf.L - l], 
                                                            [-new_bf.L + l, 
                                                             -self.bf.L - l]])
        lambda_j = new_bf.eigenvalues()
        P = jnp.diag(jax.vmap(self.prior.kernel.spectral_density, 0, 0)(jnp.sqrt(lambda_j)))
        m, S = project_regularized(self.bf, 
                                     new_bf, 
                                     self.mean_parameters, 
                                     P,
                                     **kwargs)
        alpha, B = mean_to_dual(m, 
                                S, 
                                new_bf, 
                                self.prior.kernel.spectral_density, 
                                self.likelihood.obs_stddev)
        return self.replace(alpha=alpha, B=B, bf=new_bf)

    def change_basis_mm(self, new_bf, inputs):
        """Basis change of the HGP through moment matching at discrete points.

        Parameters
        ----------
        new_bf : fasthgp.kernels.LaplaceBF
            Basis to change to
        inputs : array_like
            Inputs at which to do moment matching.

        Returns
        -------
        HGP
           An HGP with a new basis.
        """        
        qu_new = project_mm(self.bf, new_bf, self.mean_parameters, inputs)
        alpha, B = mean_to_dual(qu_new.mean(), 
                                qu_new.covariance(), 
                                new_bf, 
                                self.prior.kernel.spectral_density, 
                                self.likelihood.obs_stddev)
        return self.replace(alpha=alpha, B=B, bf=new_bf)
    
    def change_basis_regularized_mm(self, new_bf, D1, D2):
        """Basis change of the HGP through moment matching at discrete points.

        Regularizes the basis change 

        Parameters
        ----------
        new_bf : fasthgp.kernels.LaplaceBF
            Basis to change to
        D1 : array_like
            Inputs at which to do moment matching toward previous posterior.
        D2 : array_like
            Inputs at which to do moment matching toward the prior.
            

        Returns
        -------
        HGP
           An HGP with a new basis.
        """        
        qu_new = project_regularized_mm(
                                    self.bf, 
                                    new_bf, 
                                    self.mean_parameters, 
                                    self.prior.kernel.spectral_density, 
                                    D1, 
                                    D2)
        alpha, B = mean_to_dual(qu_new.mean(), 
                                qu_new.covariance(), 
                                new_bf, 
                                self.prior.kernel.spectral_density, 
                                self.likelihood.obs_stddev)
        return self.replace(alpha=alpha, B=B, bf=new_bf)

    @property
    def mean_parameters(self):
        """Mean parameters of the HGP

        Returns
        -------
        (Array, Array)
            Mean parametrization of the HGP (m, S)
        """        
        return dual_to_mean(self.alpha, self.B, self.bf, self.prior.kernel.spectral_density, self.likelihood.obs_stddev)

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import cola
from gpjax.gaussian_distribution import GaussianDistribution
from .utils import integrate


# def integrate(fun, limits, N=100, args=[]):
#     # Chebyshev-Gauss integration (second kind)
#     i = jnp.arange(1, N+1)
#     xi = jnp.cos(i/(N+1) * jnp.pi)
#     wi = jnp.pi/(N+1)*jnp.sin(i/(N+1) * jnp.pi)
#     a, b = limits
#     scale = (b - a)/2
#     return scale * wi @ fun(scale * xi + (a + b)/2, *args)


def project(old_bf, new_bf, qu, **kwargs):
        old_n, new_n = old_bf.M, new_bf.M
        # Integration limits -- move in from the boundary if not explicitly given
        l = old_bf.L/10
        limits = kwargs.get("limits", [-old_bf.L + l, old_bf.L - l])
        c1 = lambda x, i, j: old_bf._1d_call(x, i) * new_bf._1d_call(x, j)
        c2 = lambda x, i, j: new_bf._1d_call(x, i) * new_bf._1d_call(x, j)
        i = jnp.arange(1, old_n + 1)
        j = jnp.arange(1, new_n + 1)
        N = kwargs.get("N", 100)
        g1 = lambda i, j: integrate(c1, limits, N=N, args=(i, j))
        g2 = lambda i, j: integrate(c2, limits, N=N, args=(i, j))
        C1 = jax.vmap(jax.vmap(g1, (0, None), 0), (None, 0), 0)(i, j)
        C2 = jax.vmap(jax.vmap(g2, (0, None), 0), (None, 0), 0)(j, j)
        m, S = qu.mean(), qu.covariance()
        
        T = jnp.linalg.lstsq(C2, C1)[0]
        mp = T @ m
        Sp = T @ S @ T.T
        return GaussianDistribution(loc=jnp.atleast_1d(mp.squeeze()),
                                    scale=cola.ops.Dense(Sp))

def project_regularized(old_bf, new_bf, qu, P, **kwargs):
        old_n, new_n = old_bf.M, new_bf.M
        # Integration limits -- move in from the boundary if not explicitly given
        l = old_bf.L/10
        om_limits = jnp.array(kwargs.get("limits", [-old_bf.L + l, old_bf.L - l]))
        p_limits = jnp.array(kwargs.get("prior_limits", [[old_bf.L + l, 
                                                          new_bf.L - l],
                                                        [-new_bf.L + l, 
                                                        -old_bf.L - l]]))
        c1 = lambda x, i, j: old_bf._1d_call(x, i) * new_bf._1d_call(x, j)
        c2 = lambda x, i, j: new_bf._1d_call(x, i) * new_bf._1d_call(x, j)
        i = jnp.arange(1, old_n + 1)
        j = jnp.arange(1, new_n + 1)
        N = kwargs.get("N", 100)
        g1 = lambda i, j, limits: integrate(c1, limits, N=N, args=(i, j))
        g2 = lambda i, j, limits: integrate(c2, limits, N=N, args=(i, j))
        vg2 = jax.vmap(jax.vmap(g2, (0, None, None), 0), (None, 0, None), 0)
        C1_om = jax.vmap(jax.vmap(g1, (0, None, None), 0), (None, 0, None), 0)(i, j, om_limits)
        C2_om = vg2(j, j, om_limits)
        C2_p = jnp.zeros_like(C2_om)
        for lim in p_limits:
            C2_p += vg2(j, j, lim)

        K21 = C2_om.T @ C1_om
        K2om = C2_om @ C2_om.T
        K2p = C2_p @ C2_p.T
        R, _ = jsp.linalg.cho_factor(K2om + K2p, lower=True)
        T = jsp.linalg.cho_solve((R, True), jnp.identity(R.shape[0]))

        m, S = qu.mean(), qu.covariance()

        mp = T @ K21 @ m
        M = K21 @ S @ K21.T + K2p @ P @ K2p.T
        Sp = T @ M @ T.T
        return GaussianDistribution(loc=jnp.atleast_1d(mp.squeeze()),
                                    scale=cola.ops.Dense(Sp))

def project_mm(old_bf, 
            new_bf, 
            qu, 
            inputs):
    """Projection between basis functions potentially defined on different domains.

    Parameters
    ----------
    old_bf : Callable
        Old basis functions.
    new_bf : Callable
        New basis functions.
    qu : GaussianDistribution
        Variational distribution on old basis.
    inputs : Float[Array, "N D"]
        Inputs on which to draw the new posterior toward the old posterior.
    
    Returns
    -------
    GaussianDistribution
        Variational distributon on the new basis.
    """    
    m, S = qu.mean(), qu.covariance()
    newPhi = new_bf(inputs).T
    oldPhi = old_bf(inputs).T
    Pnewnew = newPhi @ newPhi.T
    Pnewold = newPhi @ oldPhi.T
    P = jnp.linalg.solve(Pnewnew, Pnewold)
    mt = P @ m
    St = P @ S @ P.T
    return GaussianDistribution(loc=jnp.atleast_1d(mt.squeeze()),
                                scale=cola.ops.Dense(St))

def project_regularized_mm(old_bf, 
                           new_bf, 
                           qu, 
                           spectral_density, 
                           D1, 
                           D2):
    """Regularized projection between basis functions potentially defined on different domains.

    Parameters
    ----------
    old_bf : Callable
        Old basis functions.
    new_bf : Callable
        New basis functions.
    qu : GaussianDistribution
        Variational distribution on old basis.
    spectral_density : Callable
        Spectral density of the kernel being used
    D1 : Float[Array, "N D"]
        Inputs on which to draw the new posterior toward the old posterior.
    D2 : Float[Array, "N D"]
        Inputs on which to draw the new posterior toward the prior.

    Returns
    -------
    GaussianDistribution
        Variational distributon on the new basis.
    """    

    m, S = qu.mean(), qu.covariance()
    lambda_j = new_bf.eigenvalues()
    Lambda = jnp.diag(spectral_density(jnp.sqrt(lambda_j)))
    
    Phi22 = new_bf(D2).T
    Phi21 = new_bf(D1).T
    Phi1 = old_bf(D1).T
    P = Phi22.T @ Lambda @ Phi22

    mu1 = Phi1.T @ m
    
    K1 = jnp.matmul(Phi21, Phi21.T)
    K2 = jnp.matmul(Phi22, Phi22.T)
    K1K2inv = jnp.linalg.solve(K1 + K2, jnp.identity(K1.shape[0]))
    m2 = K1K2inv @ (Phi21 @ mu1)
    Sigma1 = Phi1.T @ S @ Phi1
    M = Phi21 @ Sigma1 @ Phi21.T + Phi22 @ P @ Phi22.T
    S2 = K1K2inv @ M @ K1K2inv
    return GaussianDistribution(loc=jnp.atleast_1d(m2.squeeze()),
                                scale=cola.ops.Dense(S2))

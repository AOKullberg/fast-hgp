from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy as jsp

def update_with_batch(qu, 
                      bf, 
                      data, 
                      noise_cov):
    """Update a given posterior with a new batch of data.

    Parameters
    ----------
    qu : GaussianDistribution
        The posterior in mean parametrization
    bf : Callable
        The basis functions as a callable object.
    data : gpjax.Dataset
        A dataset to be processed.
    noise_cov : Float
        The noise covariance.

    Returns
    -------
    GaussianDistribution
        The new posterior in mean parametrization
    """    
    alpha, B = mean_to_dual(qu.mean(), qu.covariance(), bf, noise_cov)
    alphai, Bi = compute_dual(bf, data)
    alpha_new = alpha.squeeze() + alphai.squeeze()
    B_new = B.squeeze() + Bi.squeeze()
    return dual_to_mean(alpha_new, B_new, bf, noise_cov)

def compute_dual(bf, data):
    """Compute dual representation of the given data.

    Parameters
    ----------
    bf : Callable
        The basis functions as a callable object.
    data : gpjax.Dataset
        A dataset to be processed.

    Returns
    -------
    (Float[Array, "M L"], Float[Array, "L M M"])
        First and second order dual parameters of the posterior given the data.
    """    
    X, y = data.X, data.y
    Phi = bf(X)
    B = jnp.matmul(Phi.T, Phi)
    alpha = jnp.matmul(Phi.T, y)
    return (alpha.squeeze(), B)

@partial(jax.jit, static_argnums=(3,))
def mean_to_dual(m, 
                 S, 
                 bf, 
                 spectral_density, 
                 noise_cov):
    """Move from mean to dual representation of a posterior on bf weights.

    Go from mean parameters $(m, S)$ to dual parameters $(\alpha, B)$, i.e.,
    $(m, S) \to (\alpha, B)$ of the variational distribution on the basis 
    function weights.

    Parameters
    ----------
    m : Float[Array, "M"]
        Mean of the posterior.
    S : Float[Array, "M M"]
        Covariance of the posterior.
    bf : Callable
        The basis functions as a callable object.
    spectral_density : Callable
        Spectral density of the kernel being used
    noise_cov : Float
        Noise covariance.

    Returns
    -------
    GaussianDistribution
        Dual parametrization of the posterior
    """    
    jitter = 1e-6
    SR, _ = jsp.linalg.cho_factor(S + jnp.identity(S.shape[-1]) * jitter, lower=True)
    alpha = noise_cov * jsp.linalg.cho_solve((SR, True), m)

    SRinv = jsp.linalg.cho_solve((SR, True), jnp.identity(SR.shape[-1]))
    lambda_j = bf.eigenvalues()
    Lambdainv = jnp.diag(1/jax.vmap(spectral_density, 0, 0)(jnp.sqrt(lambda_j)))
    # Lambdainv = jnp.diag(1/jnp.prod(jnp.atleast_2d(spectral_density(jnp.sqrt(lambda_j))), axis=0))
    B = noise_cov * (SRinv - Lambdainv)
    return (alpha, B)

@partial(jax.jit, static_argnums=(3,))
def dual_to_mean(alpha, 
                 B, 
                 bf, 
                 spectral_density, 
                 noise_cov):
    """Move from dual to mean representation of a posterior on bf weights.

    Go from dual parameters $(\alpha, B)$ to mean parameters $(m, S)$, i.e.,
    $(\alpha, B) \to (m, S)$ of the variational distribution on the basis 
    function weights.

    Parameters
    ----------
    alpha : Float[Array, "M L"]
        First order dual parameter
    B : Float[Array, "L M M"]
        Second order dual parameter
    bf : Callable
        The basis functions as a callable object.
    spectral_density : Callable
        Spectral density of the kernel being used
    noise_cov : Float
        Noise covariance.

    Returns
    -------
    GaussianDistribution
        Mean parametrization of the posterior
    """    
    lambda_j = bf.eigenvalues()
    Lambdainv = jnp.diag(1/jax.vmap(spectral_density, 0, 0)(jnp.sqrt(lambda_j)))
    # Lambdainv = jnp.diag(1/jnp.prod(jnp.atleast_2d(spectral_density(jnp.sqrt(lambda_j))), axis=0))
    Sigmainv = B + noise_cov * Lambdainv
    SR, _ = jsp.linalg.cho_factor(Sigmainv, lower=True)
    Sigma = jsp.linalg.cho_solve((SR, True), jnp.identity(SR.shape[-1]))
    m = Sigma @ alpha
    S = noise_cov * Sigma
    return (jnp.atleast_1d(m.squeeze()), S)

@jax.jit
def predict(mean,
            covariance,
            bf, 
            test_inputs):
    """Predict at new test locations

    Parameters
    ----------
    qu : GaussianDistribution
        Varitional distribution on the basis function weights.
    bf : Callable
        The basis functions as a callable object.
    test_inputs : Float[Array, "N D"]
        Inputs at which to predict the function values.

    Returns
    -------
    GaussianDistribution
        Posterior on the test inputs.
    """    
    Phi = bf(test_inputs).T
    mean = Phi.T @ mean
    cov = Phi.T @ covariance @ Phi
    return mean, cov

vpredict = jax.jit(jax.vmap(lambda m, S, bf, x: predict(m, S, bf, x[None, :]), (None, None, None, 0), (0, 0)))
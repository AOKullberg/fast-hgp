from itertools import combinations_with_replacement
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
from jaxtyping import (
    Float,
    Num,
)
from gpjax.typing import (
    Array,
    ScalarFloat
)
from beartype.typing import (
    Callable,
)
import pickle


# Theta computation for given input, index and domain size
theta = lambda x, i, L: jnp.pi * i * (x + L) / (2 * L) - jnp.pi / 2
# Gamma computation for a **particular** k
@jax.jit
def gamma_k(x: Num[Array, "D"], 
            kis: Num[Array, "D"], 
            Ls: Num[Array, "D"]) -> ScalarFloat:
    """Computes a particular gamma entry.

    Parameters
    ----------
    x : Input
    kis : Indices of this entry of gamma.
    Ls : Domain sizes along each dimension

    Returns
    -------
        Gamma_kis
    """ 
    return jnp.prod(jnp.sin(theta(x, kis, Ls)))
    
# Gamma vectorized over different k
vkgamma = jax.jit(jax.vmap(gamma_k, (None, 0, None), 0))
# Gamma vectorized over x and k
vxgamma = jax.jit(jax.vmap(vkgamma, (0, None, None), 0))

@jax.jit
def gamma(x: Num[Array, "N D"], 
          ks: Num[Array, "M D"], 
          Ld: Num[Array, "D"]) -> Float[Array, "L"]:
    """Computes gamma for the laplace bf on a rectangular domain

    Parameters
    ----------
    x : Inputs for which to compute gamma
    ks : Indices for gamma
    Ld : Domain sizes in the different dimensions

    Returns
    -------
        Gamma for given indices
    """    
    return jnp.prod(1 / (2 * Ld)) * jnp.sum(vxgamma(x, ks, Ld), axis=0)

def ind(k: Num[Array, "D"], 
        md: Num[Array, "D"]) -> Num[Array, "L"]:
    """Computes indices into a Gamma on vector format

    Parameters
    ----------
    k : Basis function indices to compute Gamma indices for
    md : Number of basis functions along each dimension

    Returns
    -------
        Indices into vector Gamma.
    """    
    # Inefficient code (easier to follow probably)
    # index = 0
    # D = len(md)
    # for i in range(D-1):
    #     index += 3**(D-(i+1)) * jnp.prod(md[i+1:]) * (k[i] - (1 - md[i]))
    # return index + (k[-1] - (1 - md[-1]))
    tmp = jnp.flip(jnp.cumprod(3*md[1:]))
    c = jnp.concatenate([tmp, jnp.array([1])])
    return c @ (k - (1 - md))

@jax.jit
def T_gamma(Gamma: Num[Array, "m1m2...mD"], 
             md: Num[Array, "D"], 
             i: Num[Array, "D"], 
             j: Num[Array, "D"], 
             p: Num[Array, "2^D"]) -> ScalarFloat:
    """Computes a particular entry of the precision matrix given Gamma as a vector.

    The basis functions are indexed along each dimension. i and j are vectors of indices along each dimension for the particular basis functions of interest.

    Parameters
    ----------
    Gamma : Gamma formatted as a vector.
    md : Number of basis functions along each dimension
    i : Indices of the basis function along each dimension
    j : Indices of the (other) basis function along each dimension
    p : Possible permutations

    Returns
    -------
        The particular entry of the precision matrix defined by i and j
    """    
    # Given a particular i and j, extract the relevant Gammas
    k = i + p * j
    c = jnp.prod(p, axis=1)
    indices = jax.vmap(ind, (0, None), 0)(k, md)
    return c @ Gamma[indices.astype(int)]

@jax.jit
def B(Gamma: Num[Array, "m1m2...mD"], 
       indices: Num[Array, ""], 
       md: Num[Array, "D"], 
       p: Num[Array, "L"]) -> Float[Array, "M M"]:
    """Reconstructs the precision matrix given Gamma as a vector.

    Parameters
    ----------
    Gamma : Gamma formatted as a vector.
    indices : Array of indices of the basis functions
    md : Number of basis functions along each dimension
    p : Possible permutations

    Returns
    -------
        Precision matrix for the given indices and Gamma
    """    
    return jax.vmap(jax.vmap(T_gamma, (None, None, 0, None, None), 0),
                    (None, None, None, 0, None), 0)(Gamma, md, indices, indices, p)

# Computes upper triangular part of the information/dual matrix
@jax.jit
def B_triu(Gamma: Num[Array, "m1 m2 ... mD"], 
            indices: Num[Array, ""], 
            md: Num[Array, "D"], 
            p: Num[Array, "L"]) -> Float[Array, "M"]:
    """Computes the upper triangular part of the precision matrix.

    NB: This is inefficient -- combine_with_rep uses a lot of memory.

    Parameters
    ----------
    Gamma : Gamma formatted as a vector.
    indices : Array of indices of the basis functions
    md : Number of basis functions along each dimension
    p : Possible permutations

    Returns
    -------
        Upper triangular part of the precision matrix for the given indices and Gamma
    """      
    # Creates an array with all possible permutations between indices (since I is symmetric we only need to compute the upper triangular part)
    # Assume a matrix A=[a1; a2; a3], combinations_with_replacement creates all possible combinations of a1, a2, a3, assuming that the combinations
    # are "symmetric". I.e., combinations_with_replacement(A) = [(a1, a1), (a1, a2), (a1, a3), (a2, a2), (a2, a3), (a3, a3)].
    # E.g., (a2, a1) is ignored because (a1, a2) exists.
    # Faster implementation
    m = int(indices.shape[0]/2 * (2*indices[0,0] + (indices.shape[0] - 1)))
    i, j = combine_with_rep(indices, indices, m).T
    # Legacy implementation
    # tmp = jnp.array(list(combinations_with_replacement(indices, 2)))
    # i, j = tmp[:, 0], tmp[:, 1] # Extract the two index matrices
    return jax.vmap(T_gamma, (None, None, 0, 0, None), 0)(Gamma, md, i, j, p)

@jax.jit
def B_diag(Gamma: Num[Array, "m1m2...mD"], 
            indices: Num[Array, ""], 
            md: Num[Array, "D"], 
            p: Num[Array, "L"]) -> Float[Array, "M"]:
    """ Reconstructs the diagonal of the precision matrix.

    Parameters
    ----------
    Gamma : Gamma formatted as a vector.
    indices : Array of indices of the basis functions
    md : Number of basis functions along each dimension
    p : Possible permutations

    Returns
    -------
        Diagonal part of the precision matrix for the given indices and Gamma
    """    
    return jax.vmap(T_gamma, (None, None, 0, 0, None), 0)(Gamma, md, indices, indices, p)


@jax.jit
def TT_gamma(Gamma: Num[Array, "m1 m2 ... mD"], 
             md: Num[Array, "D"], 
             i: Num[Array, "D"], 
             j: Num[Array, "D"], 
             p: Num[Array, "2^D"]) -> ScalarFloat:
    """Computes a particular entry of the precision matrix given Gamma as a tensor.

    The basis functions are indexed along each dimension. i and j are vectors of indices along each dimension for the particular basis functions of interest.

    Parameters
    ----------
    Gamma : Gamma formatted as a tensor.
    md : Number of basis functions along each dimension
    i : Indices of the basis function along each dimension
    j : Indices of the (other) basis function along each dimension
    p : Possible permutations

    Returns
    -------
        The particular entry of the precision matrix defined by i and j
    """    
    # Given a particular i and j, extract the relevant Gammas
    k = i + p * j
    c = jnp.prod(p, axis=1)
    # Slicing doesn't work for JIT:ed things -- the slices can't be calculated unfortunately
    # slices = get_slice(k - (1 - md))
    # return c @ Gamma_t[slices].flatten()
    return c @ Gamma[*(k - (1 - md)).T.astype(int)]

@jax.jit
def TB(Gamma: Num[Array, "m1 m2 ... mD"], 
       indices: Num[Array, ""], 
       md: Num[Array, "D"], 
       p: Num[Array, "L"]) -> Float[Array, "M M"]:
    """Reconstructs the precision matrix given Gamma as a tensor.

    Parameters
    ----------
    Gamma : Gamma formatted as a tensor.
    indices : Array of indices of the basis functions
    md : Number of basis functions along each dimension
    p : Possible permutations

    Returns
    -------
        Precision matrix for the given indices and Gamma
    """    
    return jax.vmap(jax.vmap(TT_gamma, (None, None, 0, None, None), 0),
                    (None, None, None, 0, None), 0)(Gamma, md, indices, indices, p)

def TB_triu(Gamma: Num[Array, "m1 m2 ... mD"], 
            indices: Num[Array, ""], 
            md: Num[Array, "D"], 
            p: Num[Array, "L"]) -> Float[Array, "M"]:
    """Computes the upper triangular part of the precision matrix.

    NB: This is extremely inefficient -- combinations_with_replacement is extremely slow. Computational speed could be remedied by use of meshgrid but this requires a lot of storage.

    Parameters
    ----------
    Gamma : Gamma formatted as a tensor.
    indices : Array of indices of the basis functions
    md : Number of basis functions along each dimension
    p : Possible permutations

    Returns
    -------
        Upper triangular part of the precision matrix for the given indices and Gamma
    """    
    # Creates an array with all possible permutations between indices (since I is symmetric we only need to compute the upper triangular part)
    # Assume a matrix A=[a1; a2; a3], combinations_with_replacement creates all possible combinations of a1, a2, a3, assuming that the combinations
    # are "symmetric". I.e., combinations_with_replacement(A) = [(a1, a1), (a1, a2), (a1, a3), (a2, a2), (a2, a3), (a3, a3)].
    # E.g., (a2, a1) is ignored because (a1, a2) exists.
    tmp = jnp.array(list(combinations_with_replacement(indices, 2)))
    i, j = tmp[:, 0], tmp[:, 1] # Extract the two index matrices
    return jax.vmap(TT_gamma, (None, None, 0, 0, None), 0)(Gamma, md, i, j, p)

@jax.jit
def TB_diag(Gamma: Num[Array, "m1 m2 ... mD"], 
            indices: Num[Array, ""], 
            md: Num[Array, "D"], 
            p: Num[Array, "L"]) -> Float[Array, "M"]:
    """ Reconstructs the diagonal of the precision matrix.

    Parameters
    ----------
    Gamma : Gamma formatted as a tensor.
    indices : Array of indices of the basis functions
    md : Number of basis functions along each dimension
    p : Possible permutations

    Returns
    -------
        Diagonal part of the precision matrix for the given indices and Gamma
    """    
    return jax.vmap(TT_gamma, (None, None, 0, 0, None), 0)(Gamma, md, indices, indices, p)

def combine_with_rep(a, b, m):
    """ Fast implementation of combinations_with_replacement -- can **only** combine vectors of the same size
     Unfortunately creates a quite big matrix as a substep, which takes time and storage.

    Parameters
    ----------
    a : Elements to combine
    b : (Other) elements to combine
    m : Number of unique elements in b

    Returns
    -------
        An array of combined elements from a and b
    """    
    n = a.shape[0]
    R = jnp.repeat(a, jnp.arange(n, 0, -1), axis=0, total_repeat_length=m)
    S = jnp.tile(b[None, :], (n, 1, 1))[*jnp.triu_indices(n), :]
    return jnp.vstack([R[None,...], S[None,...]])


def sym(triu: Num[Array, "N"]):
    """Creates symmetric matrix given upper triangular part of matrix.

    Parameters
    ----------
    triu : Upper triangular part of matrix.

    Returns
    -------
        Reconstructed symmetric matrix
    """    
    m = int(jnp.sqrt(2*triu.shape[0] + 1/4) - 1/2)
    K = jnp.zeros((m, m))
    K = K.at[jnp.triu_indices(m)].set(triu)
    K += K.T
    return K - jnp.diag(K.diagonal()) / 2

def integrate(fun: Callable, 
              limits: list, 
              N: int=100, 
              args: list=[]) -> ScalarFloat:
    """Chebyshev-Gauss integration (second kind)

    Parameters
    ----------
    fun : Function to integrate, 0th argument is integrated w.r.t. to
    limits : Limits of the integration
    N : Number of polynomials to use
    args : Additional arguments to function.

    Returns
    -------
        Integration value
    """    
    i = jnp.arange(1, N+1)
    xi = jnp.cos(i/(N+1) * jnp.pi)[:, None]
    wi = jnp.pi/(N+1)*jnp.sin(i/(N+1) * jnp.pi)
    a, b = limits
    scale = (b - a)/2
    return jnp.prod(scale) * wi @ fun(scale * xi + (a + b)/2, *args)

def save_model(model, filename):
    """Saves the model in two files. 
    
    The Pytree definition is saved in a pickle file
    The Pytree values are saved in an .npz format

    NB: Give the filename **without** extension.

    Parameters
    ----------
    model : Pytree to save
    filename : String
    """    
    values, tree = tree_flatten(model)
    jnp.savez(filename + "_values", *values)
    with open(filename + ".pickle", "wb") as file:
        pickle.dump(tree, file)

def load_model(filename):
    """Load the model from the given filename

    NB: Give the filename **without** extension.

    Parameters
    ----------
    filename : String

    Returns
    -------
        The model defined in the file given (if it exists)
    """    
    values = list(jnp.load(filename + "_values.npz").values())
    with open(filename + ".pickle", "rb") as file:
        tree = pickle.load(file)
    return tree_unflatten(tree, values)
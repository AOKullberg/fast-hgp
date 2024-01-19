from itertools import combinations_with_replacement
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
import pickle

@jax.jit
def gamma(x, ks, Ld):
    # Compute all possible gammas
    # return jnp.prod(1/Ld) * jnp.sum(vxgamma(x, ks, Ld), axis=0)
    return jnp.prod(1 / (2 * Ld)) * jnp.sum(vxgamma(x, ks, Ld), axis=0)

@jax.jit
def TT_gamma(Gamma, md, i, j, p):
    # Assumes tensor format of Gamma
    # Given a particular i and j, extract the relevant Gammas
    k = i + p * j
    c = jnp.prod(p, axis=1)
    # Slicing doesn't work for JIT:ed things -- the slices can't be calculated unfortunately
    # slices = get_slice(k - (1 - md))
    # return c @ Gamma_t[slices].flatten()
    return c @ Gamma[*(k - (1 - md)).T.astype(int)]

@jax.jit
def TB(Gamma, indices, md, p):
    return jax.vmap(jax.vmap(TT_gamma, (None, None, 0, None, None), 0),
                    (None, None, None, 0, None), 0)(Gamma, md, indices, indices, p)

def TB_triu(Gamma, indices, md, p):
    # Creates an array with all possible permutations between indices (since I is symmetric we only need to compute the upper triangular part)
    # Assume a matrix A=[a1; a2; a3], combinations_with_replacement creates all possible combinations of a1, a2, a3, assuming that the combinations
    # are "symmetric". I.e., combinations_with_replacement(A) = [(a1, a1), (a1, a2), (a1, a3), (a2, a2), (a2, a3), (a3, a3)].
    # E.g., (a2, a1) is ignored because (a1, a2) exists.
    tmp = jnp.array(list(combinations_with_replacement(indices, 2)))
    i, j = tmp[:, 0], tmp[:, 1] # Extract the two index matrices
    return jax.vmap(TT_gamma, (None, None, 0, 0, None), 0)(Gamma, md, i, j, p)

@jax.jit
def TB_diag(Gamma, indices, md, p):
    return jax.vmap(TT_gamma, (None, None, 0, 0, None), 0)(Gamma, md, indices, indices, p)

@jax.jit
def T_gamma(Gamma, md, i, j, p):
    # Given a particular i and j, extract the relevant Gammas
    k = i + p * j
    c = jnp.prod(p, axis=1)
    indices = jax.vmap(ind, (0, None), 0)(k, md)
    return c @ Gamma[indices.astype(int)]

@jax.jit
def B(Gamma, indices, md, p):
    return jax.vmap(jax.vmap(T_gamma, (None, None, 0, None, None), 0),
                    (None, None, None, 0, None), 0)(Gamma, md, indices, indices, p)

def combine_with_rep(a, b, m):
    # "Fast" implementation of combinations_with_replacement -- can **only** combine vectors of the same size
    # Unfortunately creates a quite big matrix as a substep, which takes time and storage.
    n = a.shape[0]
    R = jnp.repeat(a, jnp.arange(n, 0, -1), axis=0, total_repeat_length=m)
    S = jnp.tile(b[None, :], (n, 1, 1))[*jnp.triu_indices(n), :]
    return jnp.vstack([R[None,...], S[None,...]])

# Computes upper triangular part of the information/dual matrix
@jax.jit
def B_triu(Gamma, indices, md, p):
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
def B_diag(Gamma, indices, md, p):
    return jax.vmap(T_gamma, (None, None, 0, 0, None), 0)(Gamma, md, indices, indices, p)

# Theta computation
theta = lambda x, i, L: jnp.pi * i * (x + L) / (2 * L) - jnp.pi / 2
# Gamma computation for a **particular** k
@jax.jit
def gamma_k(x, kis, Ls):
    """
    x - 1d vector -- particular data point
    Ls - a vector of Ld, d=1,...,D (1 x D)
    Kis - a vector kd, d=1,...,D (1 x D)
    """
    # D = len(kis)
    # return 1 / (2 ** D) * jnp.prod(jnp.sin(theta(x, kis, Ls)))
    return jnp.prod(jnp.sin(theta(x, kis, Ls)))
    
# Vectorized over different k
vkgamma = jax.jit(jax.vmap(gamma_k, (None, 0, None), 0))
# Vectorized over x and k
vxgamma = jax.jit(jax.vmap(vkgamma, (0, None, None), 0))

# Create a symmetric matrix given a triangular part of a matrix
def sym(triu):
    m = int(jnp.sqrt(2*triu.shape[0] + 1/4) - 1/2)
    K = jnp.zeros((m, m))
    K = K.at[jnp.triu_indices(m)].set(triu)
    K += K.T
    return K - jnp.diag(K.diagonal()) / 2

def ind(k, md):
    # Compute the indices into a 1D Gamma
    # index = 0
    # D = len(md)
    tmp = jnp.flip(jnp.cumprod(3*md[1:]))
    c = jnp.concatenate([tmp, jnp.array([1])])
    return c @ (k - (1 - md))
    # for i in range(D-1):
    #     index += 3**(D-(i+1)) * jnp.prod(md[i+1:]) * (k[i] - (1 - md[i]))
    # return index + (k[-1] - (1 - md[-1]))

def integrate(fun, limits, N=100, args=[]):
    # Chebyshev-Gauss integration (second kind)
    i = jnp.arange(1, N+1)
    xi = jnp.cos(i/(N+1) * jnp.pi)[:, None]
    wi = jnp.pi/(N+1)*jnp.sin(i/(N+1) * jnp.pi)
    a, b = limits
    scale = (b - a)/2
    return jnp.prod(scale) * wi @ fun(scale * xi + (a + b)/2, *args)

def save_model(model, filename):
    values, tree = tree_flatten(model)
    jnp.savez(filename + "_values", *values)
    with open(filename + ".pickle", "wb") as file:
        pickle.dump(tree, file)

def load_model(filename):
    values = list(jnp.load(filename + "_values.npz").values())
    with open(filename + ".pickle", "rb") as file:
        tree = pickle.load(file)
    return tree_unflatten(tree, values)
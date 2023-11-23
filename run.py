import logging
import pdb
import timeit

from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import jax

import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp

log = logging.getLogger(__name__)

def sll(q, Dtest, Dtrain):
    m, S = q.mean(), q.covariance().diagonal()
    loss_model = (0.5 * jnp.log(2 * jnp.pi * S) + (Dtest.y.flatten() - m)**2 / (2 * S)).mean()
    res = loss_model
    data_mean = Dtrain.y.mean()
    data_var = Dtrain.y.var()
    loss_trivial_model = (
        0.5 * jnp.log(2 * jnp.pi * data_var) + (Dtest.y.flatten() - data_mean)**2 / (2 * data_var)
    ).mean()
    res = res - loss_trivial_model
    return res

def nlpd(q, Dtest, likelihood):
    return jnp.mean(-likelihood.link_function(q.mean()).log_prob(Dtest.y.flatten()))

def kl(q1, q2, likelihood):
    m1, S1 = q1.mean(), q1.covariance()
    m2, S2 = q2.mean(), q2.covariance()
    k = m1.shape[0]
    R1,_ = jsp.linalg.cho_factor(S1 + jnp.identity(k)*likelihood.obs_noise, lower=True)
    R2,_ = jsp.linalg.cho_factor(S2 + jnp.identity(k)*likelihood.obs_noise, lower=True)
    tr_term = jnp.trace(jsp.linalg.cho_solve((R2, True), R1))
    log_det_term = 2 * jnp.sum(jnp.log(R2.diagonal())) - 2 * jnp.sum(jnp.log(R1.diagonal()))
    diff = m1 - m2
    quad_term = diff @ jsp.linalg.cho_solve((R2, True), diff)
    return 1/2 * (tr_term - k + quad_term + log_det_term)

def rmse(y1, y2):
    return jnp.sqrt(jnp.mean((y1.mean() - y2.mean())**2))

def eval_data(alg, data):
    log.parent.disabled = True
    alg = alg.update_with_batch(data)
    yhat, q = alg.predict(data.X)
    approx_yhat, approx_q = alg.predict(data.X, approx=True)
    t = timeit.repeat('alg.predict(data.X)', 
                      repeat=5, 
                      number=10, 
                      globals=locals())
    approx_t = timeit.repeat('alg.predict(data.X, approx=True)', 
                             repeat=5, 
                             number=10, 
                             globals=locals())
    name = type(alg).__name__
    aname = 'A' + name
    result = {
        name : dict(
        kl = 0.,
        nlpd = nlpd(yhat, data, alg.likelihood),
        sll = sll(yhat, data, data),
        rmse = 0.,
        time_mu = np.mean(t),
        time_std = np.std(t),
        m = q.mean().shape[0],
    ),
        aname : dict(
        kl = kl(yhat, approx_yhat, alg.likelihood),
        nlpd = nlpd(approx_yhat, data, alg.likelihood),
        sll = sll(approx_yhat, data, data),
        rmse = rmse(yhat, approx_yhat),
        time_mu = np.mean(approx_t),
        time_std = np.std(approx_t),
        m = approx_q.mean().shape[0],
    )}
    return result

@hydra.main(version_base=None, config_path="config", config_name="timing")
def main(cfg: DictConfig) -> None:
    jax.config.update("jax_enable_x64", True)
    log.info("Instantiating objects")
    data_generator = instantiate(cfg.example.data_generator)
    alg = instantiate(cfg.alg)
    log.info("Generating data")
    D = data_generator()
    log.info("Data generated!")
    res = eval_data(alg, D)
    log.info("Evaluation complete")
    log.info("Saving data and quitting")
    name = type(alg).__name__
    aname = 'A' + name
    np.savez('result.npz',
             **res[name])
    np.savez('aresult.npz',
             **res[aname])
    # np.savez('result.npz',
    #         state_mean=np.array(res['result'].mean),
    #         state_cov=np.array(res['result'].cov),
    #         ell=np.array(res['ell']),
    #         x=x,
    #         y=y)

if __name__ == "__main__":
    main()

import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np
import pytest

from hoag.nonlinear_least_squares import (
    _nonlinear_least_squares_loss, 
    _nonlinear_least_squares_loss_and_grad, 
    _nonlinear_least_squares_grad_hess,
)

def nls_loss(w, X, y, alpha):
    z = jnp.dot(X, w)
    sigz = jax.nn.sigmoid(z)
    loss = 0.5*jnp.sum((y - sigz)**2) + 0.5*alpha*jnp.dot(w,w)
    return loss

def hvp(f, x, v):
    return grad(lambda x: jnp.vdot(grad(f)(x), v))(x)

grad_loss = jit(grad(nls_loss))

N_SAMPLES = 100
N_FEATURES = 100
ALPHAS = [1e-4, 1e-2, 1, 1e2, 1e4]

@pytest.mark.parametrize('alpha', ALPHAS)
def test_loss(alpha):
    X = np.random.normal(size=(N_SAMPLES, N_FEATURES))
    y = np.random.binomial(1, 0.5, N_SAMPLES)
    w = np.random.normal(size=(N_FEATURES,))
    args = w, X, y, alpha
    loss_hoag = _nonlinear_least_squares_loss(*args)
    loss_jax = nls_loss(*[jnp.array(arg) for arg in args])
    np.testing.assert_allclose(loss_hoag, loss_jax, rtol=1e-6)
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

def hvp(f, w, X, y, alpha, v):
    return grad(lambda x: jnp.vdot(grad(f, argnums=0)(x, X, y, alpha), v))(w)

grad_loss = jit(grad(nls_loss, argnums=0))

N_SAMPLES = 100
N_FEATURES = 100
ALPHAS = [1e-4, 1e-2, 1, 1e2, 1e4]

def my_setup(alpha):
    X = np.random.normal(size=(N_SAMPLES, N_FEATURES))
    y = np.random.binomial(1, 0.5, N_SAMPLES)
    w = np.random.normal(size=(N_FEATURES,))
    args = w, X, y, alpha
    return args

@pytest.mark.parametrize('alpha', ALPHAS)
def test_loss(alpha):
    args = my_setup(alpha)
    loss_hoag = _nonlinear_least_squares_loss(*args)
    loss_jax = nls_loss(*[jnp.array(arg) for arg in args])
    np.testing.assert_allclose(loss_hoag, loss_jax, rtol=1e-6)

@pytest.mark.parametrize('alpha', ALPHAS)
def test_grad(alpha):
    args = my_setup(alpha)
    _, grad_hoag = _nonlinear_least_squares_loss_and_grad(*args)
    grad_jax = grad_loss(*[jnp.array(arg) for arg in args])
    np.testing.assert_allclose(grad_hoag, grad_jax, rtol=1e-3)

@pytest.mark.parametrize('alpha', ALPHAS)
def test_hessian(alpha):
    args = my_setup(alpha)
    grad_hoag, hess_fun_hoag = _nonlinear_least_squares_grad_hess(*args)
    grad_jax = grad_loss(*[jnp.array(arg) for arg in args])
    np.testing.assert_allclose(grad_hoag, grad_jax, rtol=1e-3)
    for _ in range(10):
        v = np.random.normal(size=(N_FEATURES,))
        hess_hoag = hess_fun_hoag(v)
        hess_jax = hvp(nls_loss, *args, v=v)
        np.testing.assert_allclose(hess_hoag, hess_jax, rtol=1e-2)

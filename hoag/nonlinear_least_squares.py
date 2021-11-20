import numpy as np
from scipy import sparse
from sklearn import linear_model
from scipy.special import expit
from sklearn.utils.extmath import log_logistic, safe_sparse_dot

from hoag.hoag import hoag_lbfgs


def _create_bilevel_functions(Xt, yt, Xh, yh):
    if not np.all(np.unique(yt) == np.array([-1, 1])):
        raise ValueError


    def h_func_grad(x, alpha):
        return _nonlinear_least_squares_loss_and_grad(
            x, Xt, yt, np.exp(alpha[0]))

    def h_hessian(x, alpha):
        return _logistic_grad_hess(
            x, Xt, yt, np.exp(alpha[0]))[1]

    def g_func_grad(x, alpha):
        return _nonlinear_least_squares_loss_and_grad(x, Xh, yh, 0)

    def h_crossed(x, alpha):
        return np.exp(alpha[0]) * x

    return h_func_grad, h_hessian, g_func_grad, h_crossed

class NonlinearLeastSquaresCV(linear_model._base.BaseEstimator,
                           linear_model._base.LinearClassifierMixin):

    def __init__(
                 self, alpha0=0., tol=0.1, callback=None, verbose=0,
                 tolerance_decrease='exponential', max_iter=50, shine=False, **lbfgs_kwargs):
        self.alpha0 = alpha0
        self.tol = tol
        self.callback = callback
        self.verbose = verbose
        self.tolerance_decrease = tolerance_decrease
        self.max_iter = max_iter
        self.shine = shine
        self.lbfgs_kwargs = lbfgs_kwargs

    def grid_search(self, Xt, yt, Xh, yh, callback=None, random=False):
        h_func_grad, h_hessian, g_func_grad, h_crossed = _create_bilevel_functions(
            Xt,
            yt,
            Xh,
            yh,
        )
        if random:
            grid = np.linspace(-12, 12, self.max_iter)
        else:
            grid = np.random.uniform(-12, 12, self.max_iter)
        self.coef_ = self.alpha_ = None
        min_loss = np.inf
        for cur_alpha in grid:
            if self.coef_ is None:
                x0 = np.random.randn(Xt.shape[1])
                self.coef_ = x0
                self.alpha_ = cur_alpha
            else:
                x0 = cur_coef
            if callback is not None:
                callback(self.coef_, [self.alpha_])
            opt = hoag_lbfgs(
                h_func_grad, h_hessian, h_crossed, g_func_grad, x0,
                callback=None,
                tolerance_decrease=self.tolerance_decrease,
                lambda0=np.array([cur_alpha]), maxiter=2,
                only_fit=True,
                verbose=self.verbose, shine=False, **self.lbfgs_kwargs)
            cur_coef = opt[0]
            cur_loss = _nonlinear_least_squares_loss(cur_coef, Xh, yh, 0)
            if cur_loss < min_loss:
                min_loss = cur_loss
                self.coef_ = cur_coef
                self.alpha_ = cur_alpha
        if callback is not None:
            callback(self.coef_, [self.alpha_])
        return self

    def fit(self, Xt, yt, Xh, yh, callback=None):
        x0 = np.random.randn(Xt.shape[1])
        h_func_grad, h_hessian, g_func_grad, h_crossed = _create_bilevel_functions(
            Xt,
            yt,
            Xh,
            yh,
        )
        opt = hoag_lbfgs(
            h_func_grad, h_hessian, h_crossed, g_func_grad, x0,
            callback=callback,
            tolerance_decrease=self.tolerance_decrease,
            lambda0=np.array([self.alpha0]), maxiter=self.max_iter,
            verbose=self.verbose, shine=self.shine, full_hessian=full_hessian, **self.lbfgs_kwargs)

        # opt = _minimize_lbfgsb(
        #     h_func_grad, DE_DX, H, x0, callback=callback,
        #     tolerance_decrease=self.tolerance_decrease,
        #     lambda0=self.alpha0, maxiter=self.max_iter)

        self.coef_ = opt[0]
        self.alpha_ = opt[1]
        return self

    def decision_function(self, X):
        return X.dot(self.coef_)

    def predict(self, X):
        return np.sign(self.decision_function(X))

### The following is copied from scikit-learn



# .. some helper functions for logistic_regression_path ..
def _intercept_dot(w, X):
    """Computes np.dot(X, w).
    It takes into consideration if the intercept should be fit or not.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    """
    c = 0.
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]

    z = safe_sparse_dot(X, w) + c
    return w, c, z


def _nonlinear_least_squares_loss_and_grad(w, X, y, alpha, sample_weight=None):
    """Computes the nonlinear least squares loss and gradient.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.
    Returns
    -------
    out : float
        Nonlinear least squares loss.
    grad : ndarray, shape (n_features,) or (n_features + 1,)
        Nonlinear least squares gradient.
    """
    _, n_features = X.shape
    grad = np.empty_like(w)

    w, c, z = _intercept_dot(w, X)
    sigz = expit(z)
    y_sigz = (y - sigz)

    if sample_weight is None:
        sample_weight = np.ones(y.shape[0])

    out = np.sum(sample_weight * y_sigz**2) + .5 * alpha * np.dot(w, w)

    sigsig = sigz * (1-sigz)
    z0 = sample_weight * sigsig * y_sigz
    grad[:n_features] = safe_sparse_dot(X.T, z0) + alpha * w

    # Case where we fit the intercept.
    if grad.shape[0] > n_features:
        grad[-1] = z0.sum()
    return out, grad


def _nonlinear_least_squares_loss(w, X, y, alpha, sample_weight=None):
    """Computes the nonlinear least squares loss.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.
    Returns
    -------
    out : float
        Nonlinear least squares loss.
    """
    w, c, z = _intercept_dot(w, X)

    if sample_weight is None:
        sample_weight = np.ones(y.shape[0])

    out = np.sum(sample_weight * (y - expit(z))**2) + .5 * alpha * np.dot(w, w)
    return out


def full_hessian(w, X, y, alpha):
    # XXX only for debug so no need to implement it right away
    pass


def _nonlinear_least_squares_grad_hess(w, X, y, alpha, sample_weight=None):
    """Computes the gradient and the Hessian, in the case of a nonlinear least squares loss.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.
    Returns
    -------
    grad : ndarray, shape (n_features,) or (n_features + 1,)
        Logistic gradient.
    Hs : callable
        Function that takes the gradient as a parameter and returns the
        matrix product of the Hessian and gradient.
    """
    n_samples, n_features = X.shape
    grad = np.empty_like(w)
    fit_intercept = grad.shape[0] > n_features

    w, c, z = _intercept_dot(w, X)

    if sample_weight is None:
        sample_weight = np.ones(y.shape[0])

    sigz = expit(z)
    y_sigz = (y - sigz)
    sigsig = sigz * (1-sigz)
    z0 = sample_weight * sigsig * y_sigz

    grad[:n_features] = safe_sparse_dot(X.T, z0) + alpha * w

    # Case where we fit the intercept.
    if fit_intercept:
        grad[-1] = z0.sum()

    # The mat-vec product of the Hessian
    d = sample_weight * sigsig * (y - 2*sigz*y - 2*sigz + 3*sigz**2)
    if sparse.issparse(X):
        dX = safe_sparse_dot(sparse.dia_matrix((d, 0),
                             shape=(n_samples, n_samples)), X)
    else:
        # Precompute as much as possible
        dX = d[:, np.newaxis] * X

    if fit_intercept:
        # Calculate the double derivative with respect to intercept
        # In the case of sparse matrices this returns a matrix object.
        dd_intercept = np.squeeze(np.array(dX.sum(axis=0)))

    def Hs(s):
        ret = np.empty_like(s)
        ret[:n_features] = dX.dot(s[:n_features])
        ret[:n_features] += alpha * s[:n_features]

        # For the fit intercept case.
        if fit_intercept:
            ret[:n_features] += s[-1] * dd_intercept
            ret[-1] = dd_intercept.dot(s[:n_features])
            ret[-1] += d.sum() * s[-1]
        return ret

    return grad, Hs

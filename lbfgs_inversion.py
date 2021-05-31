import time
start = time.time()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hoag.logistic import _intercept_dot, log_logistic, expit, safe_sparse_dot, sparse, np
from hoag.lbfgs import lbfgs

import warnings
warnings.filterwarnings("ignore")

plt.style.use(['science'])
plt.rcParams['figure.figsize'] = (5.5, 2.8)
plt.rcParams['font.size'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6


X, y = load_breast_cancer(return_X_y=True)
y[y==0] = -1
# create validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=3)
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=3)

dim = 30
samples = 113
n_samples = 113

def f(w):
    _, n_features = X.shape
    grad = np.empty_like(w)

    w, c, yz = _intercept_dot(w, X, y)

    # Logistic loss is the negative of the log of the logistic function.
    out = -np.sum(log_logistic(yz)) + .5 * alpha * np.dot(w, w)
    return out

def grad_f(w):
    _, n_features = X.shape
    grad = np.empty_like(w)

    w, c, yz = _intercept_dot(w, X, y)

    # Logistic loss is the negative of the log of the logistic function.
    out = -np.sum(log_logistic(yz)) + .5 * alpha * np.dot(w, w)

    z = expit(yz)
    z0 = (z - 1) * y

    grad[:n_features] = safe_sparse_dot(X.T, z0) + alpha * w
    return grad

def hess_f(w):
    n_samples, n_features = X.shape

    w, c, yz = _intercept_dot(w, X, y)

    z = expit(yz)
    z0 = (z - 1) * y

    # The mat-vec product of the Hessian
    d = z * (1 - z)
    if sparse.issparse(X):
        dX = safe_sparse_dot(sparse.dia_matrix((d, 0),
                             shape=(n_samples, n_samples)), X)
    else:
        # Precompute as much as possible
        dX = d[:, np.newaxis] * X

    H = X.T.dot(dX)
    H += alpha * np.eye(n_features)

    return H

def compute_inverse_correctness(H, hess_inv, inv_direction):
    true_inv = np.linalg.solve(H, inv_direction)
    approx_inv = hess_inv(inv_direction)
    rdiff = np.linalg.norm(true_inv - approx_inv) / np.linalg.norm(true_inv)
    ratio = np.linalg.norm(approx_inv) / np.linalg.norm(true_inv)
    correl = np.dot(true_inv, approx_inv) / (np.linalg.norm(true_inv)*np.linalg.norm(approx_inv))
    return rdiff, ratio, correl, np.linalg.norm(true_inv)


n_tries = 100
results = {
    'Additional direction': [],
    'Krylov direction': [],
    'Random direction': [],
}
X = X_train
y = y_train
alpha = 1
for i in range(n_tries):
    x0 = np.random.normal(size=(dim,))
    inverse_direction = np.random.normal(size=(dim,))
    xs, fs, hess_inv, warm_lists = lbfgs(
        x0,
        f,
        grad_f,
        hess_f,
        max_iter=1000,
        m=60,
        tol_norm=np.linalg.norm,
        tol=1e-6,
        inverse_direction_fun=lambda x: inverse_direction,
        inverse_secant_freq=5,
    )
    H = hess_f(xs[-1])
    directions = {
        'Additional direction': inverse_direction,
        # here H = F'(u*), warm_lists[0][-1] = u* - u*-1
        'Krylov direction': H.dot(warm_lists[0][-1]),
        'Random direction': np.random.normal(size=(dim,)),
    }
    for dir_name, inv_direction in directions.items():
        results[dir_name].append(
            compute_inverse_correctness(H, hess_inv, inv_direction)
        )


fig = plt.figure(figsize=(5.5 * 0.48, 2.1*0.48))
g = plt.GridSpec(1, 2, width_ratios=[0.84, .15], wspace=.3)
ax = fig.add_subplot(g[0, 0])
styles = {
    'Additional direction': dict(color='C2', marker='o'),
    'Krylov direction': dict(color='C1', marker='*'),
    'Random direction': dict(color='C0', marker='^', alpha=0.5),
}
naming = {
    'Additional direction': 'Additional',
    'Krylov direction': 'Krylov',
    'Random direction': 'Random',
}
for dir_name, dir_res in results.items():
    ax.scatter(
        # 0 rdiff, 1 ratio, 2 correl
        [m[1] for m in dir_res],
        [m[2] for m in dir_res],
        label=naming[dir_name],
        s=3.,
        **styles[dir_name],
    );
ax.set_ylim([0.994, 1.0005])
handles, labels = ax.get_legend_handles_labels()
ax.set_ylabel(r'$\operatorname{cossim}(a, b)$')
ax.set_xlabel(r'$\frac{\|a \|}{\| b \|}$')
### legend
ax_legend = fig.add_subplot(g[0, -1])
legend = ax_legend.legend(handles, labels, loc='center', ncol=1, handlelength=1.5, handletextpad=.2, title=r'\textbf{Direction}')
ax_legend.axis('off')
plt.savefig('lfbgs_inversion_opa_scatter.pdf', dpi=300);


end = time.time()
print(f'The script took {end-start} seconds to run')

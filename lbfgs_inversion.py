#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


plt.style.use(['science'])
plt.rcParams['figure.figsize'] = (5.5, 2.8)
plt.rcParams['font.size'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6


# In[4]:


plt.rcParams


# In[5]:


# # our problem is f(x) = 0.5 * x^T Ax + b^Tx
# # grad_f(x) = Ax + b
# # hess_f(x) = A
# dim = 100
# A = np.random.normal(size=[dim, dim])
# A = A.dot(A.T)
# b = np.random.normal(size=[dim])
# def f(x):
#     return 0.5 * np.dot(x, np.dot(A, x)) + np.dot(b, x)

# def grad_f(x):
#     return np.dot(A, x) + b

# def hess_f(x):
#     return A


# In[6]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


# In[7]:


# X = np.random.normal(size=[n_samples, dim])
# y = (np.random.normal(size=[n_samples]) > 0).astype(np.int)


# In[8]:


from hoag.logistic import _intercept_dot, log_logistic, expit, safe_sparse_dot, sparse, np

# dim = 10
# n_samples = 100


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


# In[9]:


from hoag.lbfgs import lbfgs


# In[10]:


def compute_inverse_correctness(H, hess_inv, inv_direction):
    true_inv = np.linalg.solve(H, inv_direction)
    approx_inv = hess_inv(inv_direction)
    rdiff = np.linalg.norm(true_inv - approx_inv) / np.linalg.norm(true_inv)
    ratio = np.linalg.norm(approx_inv) / np.linalg.norm(true_inv)
    correl = np.dot(true_inv, approx_inv) / (np.linalg.norm(true_inv)*np.linalg.norm(approx_inv))
    return rdiff, ratio, correl, np.linalg.norm(true_inv)


# lim (B - F') h = 0
# lim (I - B-1 F') h =0
# lim (F'-1 F' B-1F')h = 0
# lim (F'-1 - B-1) F'h = 0

# In[11]:


n_tries = 100
results = {
    'Additional direction': [],
    'Krylov direction': [],
    'Random direction': [],
}
# X = np.random.normal(size=[n_samples, dim])
# y = (np.random.normal(size=[n_samples]) > 0).astype(np.int)
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


# In[12]:


plt.figure(figsize=(13, 7))
metric = 2 # 0 rdiff, 1 ratio, 2 correl
ticks = []
ticks_positions = [0]
for dir_name, dir_res in results.items():
    ticks.append(dir_name)
    ticks_positions.append(ticks_positions[-1] + 1)
    plt.boxplot([1 - m[metric] for m in dir_res], positions=[ticks_positions[-1]]);
ticks_positions.pop(0)
plt.yscale('log')
plt.ylabel(r'$1 - \operatorname{cossim}(B_n^{-1}v, J_{g_{\theta}}(z^\star)^{-1}v)$')
plt.xticks(ticks_positions, ticks)
plt.savefig('lfbgs_inversion_opa_correl.png');


# In[13]:


plt.figure(figsize=(13, 7))
metric = 1 # 0 rdiff, 1 ratio, 2 correl
ticks = []
ticks_positions = [0]
for dir_name, dir_res in results.items():
    ticks.append(dir_name)
    ticks_positions.append(ticks_positions[-1] + 1)
    plt.boxplot([m[metric] for m in dir_res], positions=[ticks_positions[-1]]);
ticks_positions.pop(0)
# plt.yscale('log')
plt.ylabel(r'$\frac{\|B_n^{-1}v \|}{\| J_{g_{\theta}}(z^\star)^{-1}v \|}$')
plt.xticks(ticks_positions, ticks)
plt.savefig('lfbgs_inversion_opa_ratio.png');


# In[28]:


fig = plt.figure(figsize=(5.5 * 0.48, 2.1*0.48))
g = plt.GridSpec(1, 2, width_ratios=[0.84, .15], wspace=.3)
ax = fig.add_subplot(g[0, 0])
styles = {
    'Additional direction': dict(color='C2', marker='o'),
    'Krylov direction': dict(color='C1', marker='*'),
    'Random direction': dict(color='C0', marker='^'),
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
# ax.loglog()
# plt.ylabel(r'$\operatorname{cossim}(B_n^{-1}v, J_{g_{\theta}}(z^\star)^{-1}v)$')
ax.set_ylabel(r'$\operatorname{cossim}(a, b)$')
# plt.xlabel(r'$\frac{\|B_n^{-1}v \|}{\| J_{g_{\theta}}(z^\star)^{-1}v \|}$')
ax.set_xlabel(r'$\frac{\|a \|}{\| b \|}$')
### legend
ax_legend = fig.add_subplot(g[0, -1])
legend = ax_legend.legend(handles, labels, loc='center', ncol=1, handlelength=1.5, handletextpad=.2, title=r'\textbf{Direction}')
ax_legend.axis('off')
# plt.legend(frameon=True)
plt.savefig('lfbgs_inversion_opa_scatter.pdf', dpi=300);


# In[15]:


get_ipython().run_line_magic('debug', '')


# In[ ]:





from dataclasses import dataclass, fields, field
import time
from typing import List

from libsvmdata import fetch_libsvm
import numpy as np
import scipy.sparse as sp
from sklearn import datasets
from sklearn.model_selection import train_test_split

from hoag import LogisticRegressionCV
from hoag.logistic import _intercept_dot, log_logistic


@dataclass
class BenchResult:
    lambda_traces: List = field(default_factory=list)
    lamda_times: List = field(default_factory=list)
    beta_traces: List = field(default_factory=list)
    val_losses: List = field(default_factory=list)
    test_losses: List = field(default_factory=list)

    def __getitem__(self, field_key):
        return self.__getattribute__(field_key)

    def __setitem__(self, field_key, value):
        return self.__setattr__(field_key, value)

    def append(self, bench_res):
        for f in fields(self):
            self[f.name] += [bench_res[f.name]]

    def freeze(self):
        for f in fields(self):
            self[f.name] = np.array(self[f.name])

    def median(self):
        return BenchResult(
            *tuple(np.median(self[f.name], axis=0)
            for f in fields(self))
        )

    def quantile(self, q):
        return BenchResult(
            *tuple(np.quantile(self[f.name], q, axis=0)
            for f in fields(self))
        )


def train_test_val_split(X, y, random_state=0):
    X_train, y_train, X_test, y_test = train_test_split(
        X,
        y,
        test_size=2/3,
        random_state=random_state,
    )
    X_val, y_val, X_test, y_test = train_test_split(
        X_test,
        y_test,
        test_size=1/2,
        random_state=random_state,
    )
    return X_train, y_train, X_test, y_test, X_val, y_val

def get_20_news(random_state=0):
    # get a training set and test set
    data_train = datasets.fetch_20newsgroups_vectorized(subset='train')
    data_test = datasets.fetch_20newsgroups_vectorized(subset='test')

    X_train = data_train.data
    X_test = data_test.data
    y_train = data_train.target
    y_test = data_test.target

    # binarize labels
    y_train[data_train.target < 10] = -1
    y_train[data_train.target >= 10] = 1
    y_test[data_test.target < 10] = -1
    y_test[data_test.target >= 10] = 1

    # Regroup all
    X = sp.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])
    # Equally-sized split
    X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(
        X,
        y,
        random_state=random_state,
    )
    return X_train, y_train, X_test, y_test, X_val, y_val

def get_realsim(random_state):
    X, y = fetch_libsvm("real-sim")
    X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(
        X,
        y,
        random_state=random_state,
    )
    return X_train, y_train, X_test, y_test, X_val, y_val

def val_loss(X, y, beta):
    _, _, yz = _intercept_dot(beta, X, y)
    out = -np.sum(log_logistic(yz))
    return out

def results_for_kwargs(dataset='20news', random_state=0, search=None, **kwargs):
    if dataset == '20news':
        get_fun = get_20_news
    elif dataset == 'real-sim':
        get_fun = get_realsim
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented')
    X_train, y_train, X_test, y_test, X_val, y_val = get_fun(random_state)
    np.random.seed(random_state)
    lambda_traces = []
    lambda_times = []
    beta_traces = []
    start = time.time()
    def lambda_tracing(x, lmbd):
        delta = time.time() - start
        lambda_traces.append(np.copy(lmbd)[0])
        lambda_times.append(delta)
        beta_traces.append(x.copy())
    # optimize model parameters and hyperparameters jointly
    # using HOAG
    clf = LogisticRegressionCV(**kwargs)
    if search is None:
        clf.fit(X_train, y_train, X_test, y_test, callback=lambda_tracing)
    else:
        random = search == 'random'
        clf.grid_search(
            X_train,
            y_train,
            X_test,
            y_test,
            callback=lambda_tracing,
            random=random,
        )
    val_losses = [val_loss(X_val, y_val, beta) for beta in beta_traces]
    test_losses = [val_loss(X_test, y_test, beta) for beta in beta_traces]
    res = BenchResult(lambda_traces, lambda_times, beta_traces, val_losses, test_losses)
    return res

def randomized_results_for_kwargs(n_random_seed=10, **kwargs):
    overall_res = BenchResult()
    for seed in range(n_random_seed):
        res = results_for_kwargs(random_state=seed, **kwargs)
        overall_res.append(res)
    return overall_res

def quantized_results_for_kwargs(**kwargs):
    overall_res = randomized_results_for_kwargs(**kwargs)
    overall_res.freeze()
    median_res = overall_res.median()
    q1_res = overall_res.quantile(0.1)
    q9_res = overall_res.quantile(0.9)
    return median_res, q1_res, q9_res

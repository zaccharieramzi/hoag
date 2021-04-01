from dataclasses import dataclass, fields
import time

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from hoag import LogisticRegressionCV
from hoag.logistic import _intercept_dot, log_logistic


@dataclass
class BenchResult:
    lambda_traces = []
    lamda_times = []
    beta_traces = []
    val_losses = []
    test_losses = []

    def append(self, bench_res):
        for field in fields(self):
            self[field.name] += (bench_res[field.name])

    def freeze(self):
        for field in fields(self):
            self[field.name] = np.array(self[field.name])

    def median(self):
        return BenchResult(
            np.median(self[field.name], axis=0)
            for field in fields(self)
        )

    def quantile(self, q):
        return BenchResult(
            np.quantile(self[field.name], q, axis=0)
            for field in fields(self)
        )


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

    # create validation set
    X_test, X_val, y_test, y_val = train_test_split(
        X_test,
        y_test,
        test_size=0.5,
        random_state=random_state,
    )
    return X_train, y_train, X_test, y_test, X_val, y_val

def val_loss(X, y, beta):
    _, _, yz = _intercept_dot(beta, X, y)
    out = -np.sum(log_logistic(yz))
    return out

def results_for_kwargs(dataset='20news', random_state=0, **kwargs):
    if dataset == '20news':
        X_train, y_train, X_test, y_test, X_val, y_val = get_20_news(random_state)
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented')

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
    clf.fit(X_train, y_train, X_test, y_test, callback=lambda_tracing)
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
    median_res = overall_res.median()
    q1_res = overall_res.quantile(0.1)
    q9_res = overall_res.quantile(0.9)
    return median_res, q1_res, q9_res

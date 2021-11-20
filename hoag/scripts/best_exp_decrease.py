import numpy as np

from hoag import *
from hoag.benchmark import *


def get_alpha_for_exp_decrease(exp_decrease, max_iter=50, **kwargs):
    X_train, y_train, X_test, y_test, X_val, y_val = get_20_news(0, train_prop=0.9)
    lambda_traces = []
    lambda_times = []
    beta_traces = []
    start = time.time()
    def lambda_tracing(x, lmbd):
        delta = time.time() - start
        lambda_traces.append(np.copy(lmbd)[0])
        lambda_times.append(delta)
        beta_traces.append(x.copy())
    np.random.seed(0)
    clf = NonlinearLeastSquaresCV(
        max_iter=50, 
        maxiter_inner=1000, 
        exponential_decrease_factor=exp_decrease,
        **kwargs,
    )
    clf.fit(X_train, y_train, X_test, y_test, callback=lambda_tracing)
    val_losses = [val_loss_nls(X_val, y_val, beta) for beta in beta_traces]
    test_losses = [val_loss_nls(X_test, y_test, beta) for beta in beta_traces]
    np.save(f'val_losses_exp{exp_decrease}.npy', val_losses)
    print(exp_decrease)
    print(clf.alpha_)
    print(min(val_losses))
    return clf.alpha_, val_losses
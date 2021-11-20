import numpy as np

from hoag import *
from hoag.benchmark import *

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
clf = NonlinearLeastSquaresCV(max_iter=100, maxiter_inner=1000, shine=False, fpn=False, inner_callback=None)
clf.grid_search(X_train, y_train, X_test, y_test, callback=lambda_tracing)
val_losses = [val_loss_nls(X_val, y_val, beta) for beta in beta_traces]
test_losses = [val_loss_nls(X_test, y_test, beta) for beta in beta_traces]
print(clf.alpha_)
print(min(val_losses))
np.save('val_losses.npy', val_losses)
np.save('test_losses.npy', test_losses)
np.save('lambdas.npy', lambda_traces)
np.save('betas.npy', beta_traces)
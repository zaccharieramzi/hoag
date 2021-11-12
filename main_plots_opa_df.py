import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from hoag.benchmark import framed_results_for_kwargs

# Import common plot utils
from main_plots import SCHEME_LABELS
from main_plots import SCHEME_STYLES
from main_plots import setup_matplotlib

# Import to evaluate inversion quality in multiple direction
from sklearn.utils import check_random_state
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from hoag.lbfgs import lbfgs
from lbfgs_inversion import compute_inverse_correctness
from hoag.logistic import _intercept_dot, log_logistic, expit


SCHEMES = [
    'pure-python', 'shine-big-rank-pp', 'shine-big-rank-opa',
]

DIRECTION_STYLES = {
    'Prescribed': dict(color='C2', marker='o'),
    'Krylov': dict(color='C1', marker='*'),
    'Random': dict(color='C0', marker='^', alpha=0.5),
}

SEED = 42

FIG_WIDTH = 5.5
FIG_HEIGHT = 1.8


def evaluate_opa_inversion_quality():
    alpha = 1
    n_tries = 100

    rng = check_random_state(SEED)

    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler(with_mean=False).fit_transform(X)
    y[y == 0] = -1

    # sub sample on train set
    X, _, y, _ = train_test_split(X, y, test_size=0.8, random_state=3)
    n_samples, n_features = X.shape

    def f(w):
        w, c, yz = _intercept_dot(w, X, y)

        # Logistic loss is the negative of the log of the logistic function.
        out = -np.sum(log_logistic(yz)) + .5 * alpha * np.dot(w, w)
        return out

    def grad_f(w):
        grad = np.empty_like(w)

        w, c, yz = _intercept_dot(w, X, y)

        z = expit(yz)
        z0 = (z - 1) * y

        grad[:n_features] = X.T @ z0 + alpha * w
        return grad

    def hess_f(w):

        w, c, yz = _intercept_dot(w, X, y)

        z = expit(yz)

        # The mat-vec product of the Hessian
        d = z * (1 - z)
        dX = d[:, np.newaxis] * X

        H = X.T.dot(dX)
        H += alpha * np.eye(n_features)

        return H

    results = {
        'Prescribed': [],
        'Krylov': [],
        'Random': [],
    }
    for i in range(n_tries):
        x0 = rng.normal(size=(n_features,))
        inverse_direction = rng.normal(size=(n_features,))
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
            'Prescribed': inverse_direction,
            # here H = F'(u*), warm_lists[0][-1] = u* - u*-1
            'Krylov': H.dot(warm_lists[0][-1]),
            'Random': rng.normal(size=(n_features,)),
        }
        for dir_name, inv_direction in directions.items():
            results[dir_name].append(
                compute_inverse_correctness(H, hess_inv, inv_direction)
            )
    return results


def plot_eval_inversion(results, fig=None, g=None):

    if fig is None:
        fig = plt.figure(figsize=(FIG_WIDTH * 0.37, FIG_HEIGHT))
    grid_param = dict(nrows=2, ncols=1, height_ratios=[0.3, 0.7], hspace=0)
    if g is None:
        g = plt.GridSpec(**grid_param)
    else:
        g = g.subgridspec(**grid_param)
    ax = fig.add_subplot(g[1, 0])

    for dir_name, dir_res in results.items():
        ax.scatter(
            # 0 rdiff, 1 ratio, 2 correl
            [m[1] for m in dir_res],
            [m[2] for m in dir_res],
            label=dir_name,
            s=3.,
            **DIRECTION_STYLES[dir_name],
        )
    ax.set_xlim([0.985, 1.10])
    ax.set_ylim([0.994, 1.0005])
    ax.set_ylabel(r'$\operatorname{cossim}(a, b)$')
    ax.set_xlabel(r'${\|a \|} / {\| b \|}$')

    # Legend
    ax_legend = fig.add_subplot(g[0, 0])
    handles, labels = ax.get_legend_handles_labels()
    legend = ax_legend.legend(
        handles, labels, loc='lower center', ncol=3, handlelength=1,
        handletextpad=.2, title=r'\textbf{Direction}', markerscale=3,
        fontsize=9, title_fontsize=10, columnspacing=1
    )
    legend._legend_box.align = "left"
    ax_legend.axis('off')
    plt.savefig('lfbgs_inversion_opa_scatter.pdf', dpi=300)


def plot_results_OPA(big_df_res, fig=None, g=None):
    setup_matplotlib()

    min_per_seed = big_df_res.groupby(['seed'])['val_loss'].min() - 1e-5

    if fig is None:
        fig = plt.figure(figsize=(.63 * FIG_WIDTH, FIG_HEIGHT))
    if g is None:
        g = plt.GridSpec(1, 1)[0]

    ax = fig.add_subplot(g)

    xlim = (1000, 0)
    for scheme_label in SCHEMES:
        # as we want to modify a column, materialize the view
        df_scheme = big_df_res.query(
            f'scheme_label=="{scheme_label}"'
        ).copy()
        curve = pd.DataFrame(df_scheme.apply(
            lambda x: pd.Series({
                'seed': x['seed'],
                'time': x['time'],
                'val': x['val_loss'] - min_per_seed.loc[x['seed']]}
            ), axis=1)
        )
        t = np.logspace(-2, 3, 50)
        curve_t = (
            curve.groupby('seed').apply(
                lambda x: pd.DataFrame({
                    't': t,
                    # Linear interpolator to resample on a grid t
                    'v': interp1d(
                        x['time'], x['val'],
                        bounds_error=False,
                        fill_value=(
                            x['val'].iloc[0],
                            x['val'].iloc[-1]
                        )
                    )(t)
                }).set_index('t')
            )
        )

        # q1, q3 = .1, .9
        q1, q3 = .25, .75
        curve_t = curve_t.groupby('t')['v'].quantile(
            [0.5, q1, q3]
        ).unstack()
        lines = ax.semilogy(
            curve_t.index, curve_t[0.5],
            label=SCHEME_LABELS[scheme_label],
            linewidth=2,
            **SCHEME_STYLES[scheme_label],
        )
        median_times = df_scheme.groupby(['i_iter'])['time'].median()
        xlim = (min(xlim[0], median_times.min()),
                max(xlim[1], median_times.max()))
        ax.fill_between(
            curve_t.index, curve_t[q1], curve_t[q3],
            color=lines[0].get_color(),
            alpha=.3,
        )
    ax.set_xlim(xlim)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Test set loss')
    ax.legend()
    fig.savefig('bilevel_opa.pdf', dpi=300)
    fig.savefig(f'{results_name[:-4]}_val.pdf', dpi=300)


def plot_opa_results(big_df_res, results):
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    g = plt.GridSpec(1, 2, width_ratios=[0.63, .37],
                     wspace=.3, right=.96, left=.03)
    plot_results_OPA(big_df_res, fig=fig, g=g[0])
    plot_eval_inversion(results, fig=fig, g=g[1])

    fig.savefig('bilevel_opa.pdf', dpi=300)


if __name__ == '__main__':

    start = time.time()
    parser = argparse.ArgumentParser(
        description='Draw Figures bi-level optimization (2. and E.1.)')
    parser.add_argument('--no_recomp', '-nr', action='store_true',
                        help='No recomputation of the results.')
    parser.add_argument('--no_save', '-ns', action='store_true',
                        help='No saving of the results.')
    args = parser.parse_args()

    save_results = not args.no_save
    reload_results = not args.no_recomp
    dataset = '20news'
    maxiter_inner = 1000
    max_iter = 30
    train_prop = 90/100
    results_name = (
        f'{dataset}_mi{maxiter_inner}_tp{train_prop:.2f}_results_opa.csv'
    )

    schemes = {
        'warm-up': dict(max_iter=2, tol=0.1),
        'shine-big-rank-pp': dict(
            max_iter=max_iter, shine=True, maxcor=30,
            exponential_decrease_factor=0.78, debug=False,
            maxiter_inner=maxiter_inner, pure_python=True
        ),
        'shine-big-rank-opa': dict(
            max_iter=max_iter, shine=True, maxcor=60,
            exponential_decrease_factor=0.78, debug=False,
            maxiter_inner=maxiter_inner, pure_python=True, opa=True
        ),
        'pure-python': dict(
            max_iter=max_iter, shine=False,
            maxiter_inner=maxiter_inner, pure_python=True
        ),
        'pure-python-opa': dict(
            max_iter=max_iter, maxcor=30, shine=False,
            maxiter_inner=maxiter_inner, pure_python=True, opa=True
        ),
    }

    if reload_results:
        schemes_results = {
            scheme_label: framed_results_for_kwargs(
                train_prop=train_prop, dataset=dataset, n_random_seed=10,
                **scheme_kwargs
            ) for scheme_label, scheme_kwargs in schemes.items()
        }
        big_df_res = None
        for scheme_label, df_res in schemes_results.items():
            df_res['scheme_label'] = scheme_label
            if big_df_res is None:
                big_df_res = df_res
            else:
                big_df_res = big_df_res.append(df_res)
    else:
        big_df_res = pd.read_csv(results_name)

    results = evaluate_opa_inversion_quality()

    if save_results:
        big_df_res.to_csv(results_name)

    plot_opa_results(big_df_res, results)

    end = time.time()
    print(f'The script took {end-start} seconds to run')

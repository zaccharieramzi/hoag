import argparse
from ast import parse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from hoag.benchmark import framed_results_for_kwargs


DATASETS = ['20news']

SCHEME_LABELS = {
    'shine-big-rank-refined': r'\textbf{SHINE refine (ours)}',
    'shine-big-rank': r'\textbf{SHINE (ours)}',
    'fpn': 'Jacobian-Free',
    'original': 'HOAG',
    'grid-search': 'Grid search',
    'random-search': 'Random search',
    'truncated-inversion': 'HOAG - lim. backward',

    # Pure python labels
    'shine-big-rank-pp': r'\textbf{SHINE (ours)}',
    'shine-big-rank-opa': r'\textbf{SHINE - OPA (ours)}',
    'pure-python': 'HOAG',
    'pure-python-opa': 'HOAG - OPA',
}

SCHEME_STYLES = {
    'original': dict(color='C0', linestyle='-.'),
    'fpn': dict(color='C1', linestyle=':'),
    'shine-big-rank': dict(color='C2'),
    'shine-big-rank-refined': dict(color='C3'),
    'grid-search': dict(linestyle='--', color='C4'),
    'random-search': dict(color='C5'),
    'truncated-inversion': dict(color='C6'),

    # pure-python styles
    'pure-python': dict(color='C0', linestyle='-.'),
    'shine-big-rank-pp': dict(color='C2'),
    'shine-big-rank-opa': dict(color='chocolate'),
}

ZOOM_LIMS = {
    '20news': [  # for 20news
        45,
        600,
    ],
    'real-sim': [  # for real-sim
        30,
        1000,
    ]
}

maxiter_inner = 1000
max_iter = 50
train_prop = 90/100

schemes = {
    'warm-up': dict(max_iter=2, tol=0.1),
    'shine-big-rank': dict(
        max_iter=max_iter, shine=True, maxcor=10,
        exponential_decrease_factor=0.8, debug=True,
        maxiter_inner=maxiter_inner, nls=True,
    ),
    'fpn': dict(
        max_iter=max_iter, fpn=True, maxcor=10,
        exponential_decrease_factor=0.8, debug=True,
        maxiter_inner=maxiter_inner, nls=True,
    ),
    'original': dict(
        max_iter=max_iter, shine=False, maxiter_inner=maxiter_inner, 
        exponential_decrease_factor=0.8, nls=True),
}


def run_scheme(scheme_label, exponential_decrease_factor=None):
    scheme_kwargs = schemes[scheme_label]
    if exponential_decrease_factor is not None:
        scheme_kwargs['exponential_decrease_factor'] = exponential_decrease_factor
    framed_results_for_kwargs(
        train_prop=train_prop, dataset=DATASETS[0], n_random_seed=10,
        **schemes['warm-up']
    )
    df_res = framed_results_for_kwargs(
        train_prop=train_prop, dataset=DATASETS[0], n_random_seed=10,
        **scheme_kwargs,
    )
    df_res['scheme_label'] = scheme_label
    results_name = (
        f'{DATASETS[0]}_{scheme_label}_exp{exponential_decrease_factor}_mi{maxiter_inner}_tp{train_prop:.2f}_results.csv'
    )
    df_res.to_csv(results_name)
    return df_res

def setup_matplotlib():
    plt.style.use(['science'])
    # plt.rcParams['font.size'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['axes.labelsize'] = 10


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(
        description='Draw Figures bi-level optimization (2. and E.1.)')
    parser.add_argument('--appendix_figure', '-a', action='store_true',
                        help='Run for appendix figure.')
    parser.add_argument('--no_recomp', '-nr', action='store_true',
                        help='No recomputation of the results.')
    parser.add_argument('--no_save', '-ns', action='store_true',
                        help='No saving of the results.')
    parser.add_argument('--interp', action='store_true',
                        help='Use interpolation curves.')
    parser.add_argument('--quantile', '-q', type=int, default=10,
                        help='Use first and last q-quantile for variance.')
    parser.add_argument('--eps', type=float, default=1e-2,
                        help='Max sub-optimality level.')
    parser.add_argument('--objective', dest='subopt', action='store_false',
                        help='If set, plot the objective value instead of '
                        'the sub optimality.')
    parser.add_argument('--no_draw', '-nd', action='store_true',
                        help='Do not do the drawing.')
    args = parser.parse_args()

    save_results = not args.no_save
    reload_results = not args.no_recomp
    appendix_figure = args.appendix_figure

    for dataset in DATASETS:

        results_name = (
            f'{dataset}_mi{maxiter_inner}_tp{train_prop:.2f}_results.csv'
        )
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
            if save_results:
                big_df_res.to_csv(results_name)

    included_schemes = [
        'shine-big-rank', 'original',
        'fpn',
    ]

    if not args.no_draw:
        setup_matplotlib()
        fig = plt.figure(figsize=(5.5, 4.5))
        g = plt.GridSpec(2, 1, height_ratios=[0.4, .15], hspace=.45)
        for i, dataset in enumerate(DATASETS):
            results_name = (
                f'{dataset}_mi{maxiter_inner}_tp{train_prop:.2f}_results.csv'
            )
            big_df_res = pd.read_csv(results_name)
            min_per_seed = (
                big_df_res.groupby(['seed'])['val_loss'].min() - args.eps
            )
            if not args.subopt:
                min_per_seed *= 0
            ax = fig.add_subplot(g[i, 0])
            handles, labels = [], []
            for scheme_label in included_schemes:
                if scheme_label is None:
                    handles.append(plt.scatter([], [], alpha=0))
                    labels.append(None)
                    continue
                df_scheme = big_df_res.query(
                    f'scheme_label=="{scheme_label}"'
                ).copy()
                curve = pd.DataFrame(df_scheme.apply(
                    lambda x: pd.Series({
                        'seed': x['seed'], 'time': x['time'],
                        'i_iter': x['i_iter'],
                        'val': x['val_loss'] - min_per_seed.loc[x['seed']]
                    }), axis=1)
                )

                # q1, q3 = 1 / args.quantile, 1 - 1 / args.quantile
                q1, q3 = .25, .75
                if args.interp:
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
                            })
                        )
                    )
                    curve_t = curve_t.groupby('t')['v'].quantile(
                        [0.5, q1, q3]
                    ).unstack()
                else:
                    curve_t = (
                        curve.groupby('i_iter').quantile([0.5, q1, q3]).unstack()
                    ).set_index(('time', .5))['val']

                handles.extend(ax.semilogy(
                    curve_t.index, curve_t[0.5],
                    label=SCHEME_LABELS[scheme_label], linewidth=2,
                    **SCHEME_STYLES[scheme_label]
                ))
                ax.fill_between(
                    curve_t.index, curve_t[q1],
                    curve_t[q3], color=handles[-1].get_color(),
                    alpha=.3
                )
                labels.append(SCHEME_LABELS[scheme_label])
            ax.set_xlabel('Time (s)')
            ax.set_xlim(right=ZOOM_LIMS[dataset][0])
            ax.set_title(dataset)
            ax.set_xlim(left=0)
            # ax.set_ylim(bottom=1e-2)

        ax_legend = fig.add_subplot(g[-1, 0])
        legend = ax_legend.legend(
            handles, labels, loc='center', ncol=4,
            handlelength=1.5, handletextpad=.2
        )
        # Y label
        # fig.supylabel('Test set loss')
        ax_losses = fig.add_subplot(g[:-1], frameon=False)
        ax_losses.axes.xaxis.set_ticks([])
        ax_losses.axes.yaxis.set_ticks([])
        ax_losses.spines['top'].set_visible(False)
        ax_losses.spines['right'].set_visible(False)
        ax_losses.spines['bottom'].set_visible(False)
        ax_losses.spines['left'].set_visible(False)
        ylabel = 'Test Loss' + (' Suboptimality' if args.subopt else '')
        ax_losses.set_ylabel(ylabel, labelpad=28.)
        ax_legend.axis('off')
        fig.savefig('nls_test.pdf', dpi=300)

    end = time.time()
    print(f'The script took {end-start} seconds to run')

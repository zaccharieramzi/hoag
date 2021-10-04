import argparse
import time

import matplotlib.pyplot as plt
import pandas as pd

from hoag.benchmark import framed_results_for_kwargs


DATASETS = ['20news', 'real-sim']

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
        60,
        600,
    ],
    'real-sim': [  # for real-sim
        30,
        1000,
    ]
}


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
    args = parser.parse_args()

    save_results = not args.no_save
    reload_results = not args.no_recomp
    appendix_figure = args.appendix_figure
    maxiter_inner = 1000
    max_iter = 30
    train_prop = 90/100

    schemes = {
        'warm-up': dict(max_iter=2, tol=0.1),
        'shine-big-rank-refined': dict(
            max_iter=max_iter, shine=True, maxcor=30,
            exponential_decrease_factor=0.78, debug=True,
            maxiter_inner=maxiter_inner, refine=True, maxiter_backward=0
        ),
        'shine-big-rank': dict(
            max_iter=max_iter, shine=True, maxcor=30,
            exponential_decrease_factor=0.78, debug=True,
            maxiter_inner=maxiter_inner
        ),
        'fpn': dict(
            max_iter=max_iter, fpn=True, maxcor=30,
            exponential_decrease_factor=0.78, debug=True,
            maxiter_inner=maxiter_inner
        ),
        'original': dict(
            max_iter=max_iter, shine=False, maxiter_inner=maxiter_inner),
        'grid-search': dict(
            max_iter=10, shine=False, maxiter_inner=maxiter_inner,
            search='grid'
        ),
        'random-search': dict(
            max_iter=10, shine=False, maxiter_inner=maxiter_inner,
            search='random'
        ),
        'truncated-inversion': dict(
            max_iter=30, shine=False, maxiter_inner=maxiter_inner,
            maxiter_backward=5, refine=True
        ),
    }

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

    if not appendix_figure:
        included_schemes = [
            'shine-big-rank', 'original',
            'shine-big-rank-refined', 'fpn',
            None, 'grid-search',
        ]
    else:
        included_schemes = [
            'original', 'truncated-inversion',
            'shine-big-rank', 'shine-big-rank-refined',
            'grid-search', 'random-search',
            'fpn',
        ]

    setup_matplotlib()
    fig = plt.figure(figsize=(5.5, 6.5 if appendix_figure else 2.5))
    if appendix_figure:
        g = plt.GridSpec(3, 1, height_ratios=[0.4, 0.4, .15], hspace=.45)
    else:
        g = plt.GridSpec(2, 2, height_ratios=[0.1, .9],
                         wspace=.2, hspace=.5, top=.99, right=0.98)
    for i, dataset in enumerate(['20news', 'real-sim']):
        results_name = (
            f'{dataset}_mi{maxiter_inner}_tp{train_prop:.2f}_results.csv'
        )
        big_df_res = pd.read_csv(results_name)
        val_min_per_seed_series = big_df_res.groupby(['seed'])['val_loss'].min()
        if appendix_figure:
            ax = fig.add_subplot(g[i, 0])
        else:
            ax = fig.add_subplot(g[1, i])
        handles, labels = [], []
        for scheme_label in included_schemes:
            if scheme_label is None:
                handles.append(plt.scatter([], [], alpha=0))
                labels.append(None)
                continue
            df_scheme = big_df_res.query(f'scheme_label=="{scheme_label}"').copy()
            for seed in df_scheme['seed'].unique():
                df_scheme.loc[df_scheme['seed'] == seed, 'val_loss'] -= \
                    val_min_per_seed_series[seed]
            median_times = df_scheme.groupby(['i_iter'])['time'].median()
            groupby_val_loss = df_scheme.groupby(['i_iter'])['val_loss']
            median_val_losses = groupby_val_loss.median()
            q1_val_losses = groupby_val_loss.quantile(0.1)
            q9_val_losses = groupby_val_loss.quantile(0.9)
            handles.extend(ax.semilogy(
                median_times, median_val_losses,
                label=SCHEME_LABELS[scheme_label], linewidth=2,
                **SCHEME_STYLES[scheme_label]
            ))
            ax.fill_between(
                median_times, q1_val_losses,
                q9_val_losses, color=handles[-1].get_color(),
                alpha=.3
            )
            labels.append(SCHEME_LABELS[scheme_label])
        ax.set_xlabel('Time (s)')
        ax.set_xlim(right=ZOOM_LIMS[dataset][0])
        if i == 0 and not appendix_figure:
            ax.set_ylabel('Test set loss')
        ax.set_title(dataset)
        ax.set_xlim(left=0)

    if appendix_figure:
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
        ax_losses.set_ylabel('Test set loss', labelpad=28.)
    else:
        ax_legend = fig.add_subplot(g[0, :])
        legend = ax_legend.legend(
            handles, labels, loc='upper center', ncol=3,
            handlelength=2.5, handletextpad=1
        )
    ax_legend.axis('off')
    if appendix_figure:
        fig.savefig('bilevel_test_appendix.pdf', dpi=300)
    else:
        fig.savefig('bilevel_test.pdf', dpi=300)

    end = time.time()
    print(f'The script took {end-start} seconds to run')

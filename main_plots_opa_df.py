import argparse
import time

import matplotlib.pyplot as plt

from hoag.benchmark import framed_results_for_kwargs


import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    start = time.time()
    parser = argparse.ArgumentParser(
        description='Draw Figures bi-level optimization (2. and E.1.)')
    parser.add_argument('--no_recomp', '-nr', action='store_true',
                        help='No recomputation of the results.')
    parser.add_argument('--no_save', '-ns', action='store_true',
                        help='No saving of the results.')
    args = parser.parse_args()
    plt.rcParams['figure.figsize'] = (5.5, 2.8)
    plt.style.use(['science'])
    plt.rcParams['font.size'] = 8
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6

    save_results = not args.no_save
    reload_results = not args.no_recomp
    dataset = '20news'
    maxiter_inner = 1000
    max_iter = 30
    train_prop = 90/100
    results_name = f'{dataset}_mi{maxiter_inner}_tp{train_prop:.2f}_results_opa.csv'

    schemes = {
        'warm-up': dict(max_iter=2, tol=0.1),
        'shine-big-rank-pp': dict(max_iter=max_iter, shine=True, maxcor=30, exponential_decrease_factor=0.78, debug=False, maxiter_inner=maxiter_inner, pure_python=True),
        'shine-big-rank-opa': dict(max_iter=max_iter, shine=True, maxcor=60, exponential_decrease_factor=0.78, debug=False, maxiter_inner=maxiter_inner, pure_python=True, opa=True),
        'pure-python': dict(max_iter=max_iter, shine=False, maxiter_inner=maxiter_inner, pure_python=True),
        'pure-python-opa': dict(max_iter=max_iter, maxcor=30, shine=False, maxiter_inner=maxiter_inner, pure_python=True, opa=True),
    }


    if reload_results:
        schemes_results = {
            scheme_label: framed_results_for_kwargs(train_prop=train_prop, dataset=dataset, n_random_seed=10, **scheme_kwargs)
            for scheme_label, scheme_kwargs in schemes.items()
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


    schemes_naming = {
        'shine-big-rank-pp': r'\textbf{SHINE (ours)}',
        'shine-big-rank-opa': r'\textbf{SHINE - OPA (ours)}',
        'pure-python': 'HOAG',
        'pure-python-opa': 'HOAG - OPA',
    }

    val_min_per_seed_series = big_df_res.groupby(['seed'])['val_loss'].min()

    included_schemes = [
        'pure-python', 'shine-big-rank-pp', 'shine-big-rank-opa',
    ]

    styles = {
        'pure-python': dict(color='C0', linestyle='-.'),
        'shine-big-rank-pp': dict(
            color='C2'
        ), 'shine-big-rank-opa': dict(
            color='chocolate'
        ),
    }

    fig = plt.figure(figsize=(5.5, 2.1))
    g = plt.GridSpec(1, 2, width_ratios=[0.83, .15], wspace=.3)
    ax = fig.add_subplot(g[0, 0])
    handles, labels = [], []
    for scheme_label in included_schemes:
        df_scheme = big_df_res.query(f'scheme_label=="{scheme_label}"')
        for seed in df_scheme['seed'].unique():
            df_scheme.loc[df_scheme['seed'] == seed, 'val_loss'] -= val_min_per_seed_series[seed]
        median_times = df_scheme.groupby(['i_iter'])['time'].median()
        groupby_val_loss = df_scheme.groupby(['i_iter'])['val_loss']
        median_val_losses = groupby_val_loss.median()
        q1_val_losses = groupby_val_loss.quantile(0.1)
        q9_val_losses = groupby_val_loss.quantile(0.9)
        handles.extend(ax.semilogy(
            median_times,
            median_val_losses,
            label=schemes_naming[scheme_label],
            linewidth=2,
            **styles[scheme_label],
        ))
        plt.fill_between(
            median_times,
            q1_val_losses,
            q9_val_losses,
            color=handles[-1].get_color(),
            alpha=.3,
        )
        labels.append(schemes_naming[scheme_label])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Loss')
    ax.set_title('20news')
    ax_legend = fig.add_subplot(g[0, 1])
    ax_legend.axis('off')
    ax_legend.legend(handles, labels, loc='center', ncol=1, handlelength=1.5, handletextpad=.2)
    fig.savefig('bilevel_opa.pdf', dpi=300)
    fig.savefig(f'{results_name[:-4]}_val.pdf', dpi=300)

    end = time.time()
    print(f'The script took {end-start} seconds to run')

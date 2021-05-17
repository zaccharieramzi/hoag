#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle
from dataclasses import fields

import matplotlib.pyplot as plt
import pandas as pd

from hoag.benchmark import framed_results_for_kwargs


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


plt.rcParams['figure.figsize'] = (5.5, 2.8)
plt.style.use(['science'])
plt.rcParams['font.size'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6


# In[4]:


save_results = False
reload_results = False
dataset = '20news'
maxiter_inner = 1000
max_iter = 30
train_prop = 90/100
results_name = f'{dataset}_mi{maxiter_inner}_tp{train_prop:.2f}_results_opa.csv'


# In[5]:


results_name


# In[6]:


schemes = {
    'warm-up': dict(max_iter=2, tol=0.1),
    'shine-big-rank-pp': dict(max_iter=max_iter, shine=True, maxcor=30, exponential_decrease_factor=0.78, debug=False, maxiter_inner=maxiter_inner, pure_python=True),
    'shine-big-rank-opa': dict(max_iter=max_iter, shine=True, maxcor=60, exponential_decrease_factor=0.78, debug=False, maxiter_inner=maxiter_inner, pure_python=True, opa=True),
    'pure-python': dict(max_iter=max_iter, shine=False, maxiter_inner=maxiter_inner, pure_python=True),
    'pure-python-opa': dict(max_iter=max_iter, maxcor=30, shine=False, maxiter_inner=maxiter_inner, pure_python=True, opa=True),
}


# In[7]:


get_ipython().run_cell_magic('time', '', "if reload_results:\n    schemes_results = {\n        scheme_label: framed_results_for_kwargs(train_prop=train_prop, dataset=dataset, n_random_seed=10, **scheme_kwargs)\n        for scheme_label, scheme_kwargs in schemes.items()\n    }\n    big_df_res = None\n    for scheme_label, df_res in schemes_results.items():\n        df_res['scheme_label'] = scheme_label\n        if big_df_res is None:\n            big_df_res = df_res\n        else:\n            big_df_res = big_df_res.append(df_res)\nelse:\n    big_df_res = pd.read_csv(results_name)")


# In[8]:


if save_results:
    big_df_res.to_csv(results_name)


# In[9]:


zoom = False
if dataset == '20news':
    zoom_lims = [ # for 20news
        60,
        600,
    ]
else:
    zoom_lims = [ # for 20news
        30,
        1000,
    ]


# In[10]:


schemes_naming = {
    'shine-big-rank-pp': r'\textbf{SHINE (ours)}',
    'shine-big-rank-opa': r'\textbf{SHINE - OPA (ours)}',
    'pure-python': 'HOAG',
    'pure-python-opa': 'HOAG - OPA',
}


# In[11]:


val_min_per_seed_series = big_df_res.groupby(['seed'])['val_loss'].min()


# In[12]:


val_min_per_seed_series


# In[13]:


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
        alpha=.1 if zoom else .3,
    )
    labels.append(schemes_naming[scheme_label])
# plt.title(f'Validation loss on {dataset}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Loss')
ax.set_title('20news')
if zoom:
    ax.set_xlim(-0.1, zoom[0])
    ax.set_ylim(top=zoom[1])
ax_legend = fig.add_subplot(g[0, 1])
ax_legend.axis('off')
ax_legend.legend(handles, labels, loc='center', ncol=1, handlelength=1.5, handletextpad=.2)
fig.savefig('bilevel_opa.pdf', dpi=300)
if zoom:
    fig.savefig(f'{results_name[:-4]}_val_zoom.pdf', dpi=300)
else:
    fig.savefig(f'{results_name[:-4]}_val.pdf', dpi=300)


# In[14]:


# plt.figure(figsize=(13, 7))
# for scheme_label, scheme_results in schemes_results.items():
#     if scheme_label == 'warm-up':
#         continue
#     median_res, q1_res, q9_res = scheme_results
#     p = plt.plot(median_res.lamda_times, median_res.test_losses, label=scheme_label, linewidth=3)
#     plt.fill_between(median_res.lamda_times, q1_res.test_losses, q9_res.test_losses, color=p[0].get_color(), alpha=.3)
# plt.title(f'Test loss on {dataset}')
# plt.legend()


# In[ ]:





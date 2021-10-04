# HOAG - SHINE

This is the first part of the code for the paper "SHINE: SHaring the INverse Estimate from the forward pass for bi-level optimization and implicit models", submitted at the 2022 ICLR conference.
This source code allows to reproduce the experiments on logistic regression, i.e. Figures 1-2, and Figure E.1. in Appendix.

## General instructions

You need Python 3.7 or above to run this code.
You can then install the requirements with: `pip install -r requirements.txt`.

When running the scripts, you will see the following warning printed: `CG did not converge to the desired precision`.
It does not indicate that there is a problem with your current run.

## Reproducing Figure 1, Bi-level optimization

Figure 1. can be reproduced by running the `main_plots.py` script:

```
python main_plots.py
```

By default, the results will be re-computed and saved each time.
If you want to use the results saved from a previous run, you can use the `--no_recomp` flag.
If you want to run a test run without saving the results, you can use the `--no_save` flag.

It will take you about 2 hours to run this script in full.
It will take you about 2 seconds to run this script with saved results.


## Reproducing Figure 2., Bi-level optimization with OPA

Figure 2. can be reproduced by running the `main_plots_opa_df.py` script:

```
python main_plots_opa_df.py
```

By default, the results will be re-computed and saved each time.
If you want to use the results saved from a previous run, you can use the `--no_recomp` flag.
If you want to run a test run without saving the results, you can use the `--no_save` flag.

It will take about 1 hour to run this script in full.
It will take about 10 seconds to run this script with saved results.

## Reproducing Figure E.1., Bi-level optimization

Figure E.1. can be reproduced by running the `main_plots.py` script:

```
python main_plots.py --appendix_figure
```

By default, the results will be re-computed and saved each time.
If you want to use the results saved from a previous run, you can use the `--no_recomp` flag.
If you want to run a test run without saving the results, you can use the `--no_save` flag.

In this case, the raw results are the same as for Figure 1., so you can use these.

It will take you about 2 hours to run this script in full.
It will take you about 2 seconds to run this script with saved results.

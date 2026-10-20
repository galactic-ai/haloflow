'''
Fig. 5 of HaloFlow II: inferred vs true halo mass for all train=>test pairs
(HaloFlow, HaloFlow+DANN, HaloFlow+MMD), from the Zenodo per-galaxy summaries.

    python fig5_scatter_grid.py --zenodo <path to the Zenodo archive> [--out figures/]

Port of cell 5 of the original notebook (bin/ml4ps/cross_validation_scatter_plots.ipynb),
with the colour-blind-safe colours of the revised paper. The galaxies shown are the same
(up to 5 per halo-mass bin); the posterior quantiles come from the archived draws, so the
points differ slightly from the published figure, which used separate draws from the same models.
'''
import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from haloflow.config import setup_plotting_config
from zenodo_io import sims, load_summary, sample_indices, add_zenodo_arg, check_zenodo_arg

RED, BLUE, GREEN = '#D55E00', '#0072B2', '#009E73'
title_names = {'TNG_ALL': 'TNG', 'Eagle100': 'Eagle', 'Simba100': 'Simba'}


def errorbar(ax, summ, idx, color, label=None, marker_alpha=None, bar_alpha=0.7):
    ''' median and 16-84 percentile of the Mh posterior vs true Mh (as haloflow.npe.plotting.plot_true_pred) '''
    q16, q50, q84 = summ['Mh_p16'][idx], summ['Mh_p50'][idx], summ['Mh_p84'][idx]
    ax.plot([9.5, 14.], [9.5, 14.], c='k', ls='--', label='_nolegend_')   # drawn per method, as in plot_true_pred
    line, _, bars = ax.errorbar(summ['logMh_true'][idx], q50, yerr=[q50 - q16, q84 - q50],
                                fmt='.', color=color, ms=12, alpha=bar_alpha, label=label)
    line.set_markeredgewidth(1.5)
    if marker_alpha is not None:
        line.set_alpha(marker_alpha)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_zenodo_arg(parser)
    parser.add_argument('--out', default='figures/')
    args = parser.parse_args()
    check_zenodo_arg(parser, args)
    os.makedirs(args.out, exist_ok=True)

    setup_plotting_config()

    fig, ax = plt.subplots(len(sims), len(sims), figsize=(14, 14))
    for i, train_sim in enumerate(sims):        # columns: training simulation
        for j, test_sim in enumerate(sims):     # rows: test simulation
            sub = ax[j, i]
            hf = load_summary(args.zenodo, f'hf_{train_sim}_{test_sim}')
            idx = sample_indices(hf['logMh_true'], test_sim, 5)

            if i == j:
                errorbar(sub, hf, idx, 'gray', marker_alpha=0.55, bar_alpha=0.55)
            else:
                errorbar(sub, hf, idx, RED, label='HaloFlow', marker_alpha=0.55)
                errorbar(sub, load_summary(args.zenodo, f'hfd_{train_sim}_{test_sim}'), idx, BLUE,
                         label='HaloFlow+DANN')
                errorbar(sub, load_summary(args.zenodo, f'hfm_{train_sim}_{test_sim}'), idx, GREEN,
                         label='HaloFlow+MMD')

            sub.set_xlim(11.3, 14.)
            sub.set_ylim(11.3, 14.)
            sub.text(0.95, 0.05, r'$\textsc{%s} \Rightarrow \textsc{%s}$' % (title_names[train_sim], title_names[test_sim]),
                     transform=sub.transAxes, ha='right', va='bottom', fontsize=20)

            if j == len(sims) - 1:
                sub.set_xlabel(r"true $\log M_{\rm h}$", fontsize=20)
            else:
                sub.set_xticklabels([])
            if i == 0:
                sub.set_ylabel(r"inferred $\log M_{\rm h}$", fontsize=20)
            else:
                sub.set_yticklabels([])

    handles, labels = ax[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=22, loc='upper center', bbox_to_anchor=(0.5, 1.03), frameon=False,
               fancybox=True, handletextpad=0, ncol=3)
    fig.tight_layout()

    fig.savefig(os.path.join(args.out, 'halo_scatter_plot_npe_mags_morph_extra_test_both_dann.pdf'), bbox_inches='tight')

'''
Fig. 4 of HaloFlow II: model mismatch for HaloFlow trained on TNG. Left: inferred vs
true halo mass in domain (TNG => TNG), out of domain (TNG => EAGLE) and for the
standard method; right: DRP (TARP) coverage of the in- and out-of-domain posteriors.

    python fig4_model_mismatch.py --zenodo <path to the Zenodo archive> [--out figures/]

Port of cell 6 of the original notebook (bin/ml4ps/model_mismatch.ipynb), with the
colour-blind-safe colours of the revised paper. The galaxies shown are the same (up to 3
per halo-mass bin). The quantiles come from the archived draws and the coverage curves are
the seeded ones in coverage/, so the figure differs slightly from the published one, which
used separate draws from the same models.
'''
import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from haloflow.config import setup_plotting_config
from zenodo_io import load_summary, load_standard_method, load_coverage, sample_indices, \
        add_zenodo_arg, check_zenodo_arg

RED, BLUE = '#D55E00', '#0072B2'


def errorbar(ax, summ, idx, fmt, color, label):
    q16, q50, q84 = summ['Mh_p16'][idx], summ['Mh_p50'][idx], summ['Mh_p84'][idx]
    ax.errorbar(summ['logMh_true'][idx], q50, yerr=[q50 - q16, q84 - q50], fmt=fmt, color=color, ms=8,
                alpha=0.7, label=label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_zenodo_arg(parser)
    parser.add_argument('--out', default='figures/')
    args = parser.parse_args()
    check_zenodo_arg(parser, args)
    os.makedirs(args.out, exist_ok=True)

    setup_plotting_config()

    in_domain = load_summary(args.zenodo, 'hf_TNG_ALL_TNG_ALL')
    out_domain = load_summary(args.zenodo, 'hf_TNG_ALL_Eagle100')
    standard = load_standard_method(args.zenodo, 'TNG_ALL', 'Eagle100')
    idx_tng = sample_indices(in_domain['logMh_true'], 'TNG_ALL', 3)
    idx_eagle = sample_indices(out_domain['logMh_true'], 'Eagle100', 3)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=300, constrained_layout=True)

    # left: inferred vs true Mh
    ax[0].plot([9.5, 14.], [9.5, 14.], c='k', ls='--', label='_nolegend_')
    errorbar(ax[0], in_domain, idx_tng, '.', BLUE, 'In Domain')
    errorbar(ax[0], out_domain, idx_eagle, 'x', RED, 'Out of Domain')
    errorbar(ax[0], standard, idx_eagle, '.', 'gray', 'Standard Method')

    ax[0].set_xlabel(r"true $\log M_{\rm h}$", fontsize=20)
    ax[0].set_ylabel(r"inferred $\log M_{\rm h}$", fontsize=20)
    ax[0].set_xlim(11.3, 14.)
    ax[0].set_ylim(11.3, 14.)
    ax[0].text(0.95, 0.05, r'\textsc{TNG} $\Rightarrow$ \textsc{Eagle}', transform=ax[0].transAxes, fontsize=16,
               ha='right')

    # right: DRP coverage
    ax[1].plot([0, 1], [0, 1], ls='--', color='k')
    for run, color in [('hf_TNG_ALL_TNG_ALL', BLUE), ('hf_TNG_ALL_Eagle100', RED)]:
        alpha, ecp = load_coverage(args.zenodo, run)
        ax[1].plot(alpha, ecp, color=color)
    ax[1].set_xlabel("Credibility Level", fontsize=20)
    ax[1].set_ylabel("Expected Coverage", fontsize=20)
    ax[1].set_xlim(0., 1.)
    ax[1].set_ylim(0., 1.)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)

    fig.savefig(os.path.join(args.out, 'model_mismatch_inference.pdf'), bbox_inches='tight', dpi=300)

'''
Fig. 6 of HaloFlow II: precision (1-sigma width of the Mh posterior) and beta as a
function of true stellar mass for TNG => EAGLE, for HaloFlow+DANN, HaloFlow+MMD and
the standard method, from the Zenodo per-galaxy summaries.

    python fig6_precision_bias.py --zenodo <path to the Zenodo archive> [--out figures/]

Port of cells 5-7 of the original notebook (bin/ml4ps/precision_bias.ipynb), with the
colour-blind-safe colours of the revised paper. The notebook drew 10000 posterior samples
per galaxy; the archive holds 1000, so the curves differ slightly from the published figure.
'''
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from haloflow.config import setup_plotting_config
from zenodo_io import load_summary, load_standard_method, add_zenodo_arg, check_zenodo_arg

BLUE, GREEN = '#0072B2', '#009E73'


def precision_beta(summ):
    return 0.5 * (summ['Mh_p84'] - summ['Mh_p16']), summ['beta_Mh']


def binned_median(sm_test, values, sm_bins):
    ''' median in bins of true stellar mass, for bins with more than 25 galaxies '''
    out = np.full(len(sm_bins) - 1, np.nan)
    for i, (lo, hi) in enumerate(zip(sm_bins[:-1], sm_bins[1:])):
        mask = (sm_test >= lo) & (sm_test < hi)
        if mask.sum() > 25:
            out[i] = np.median(values[mask])
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_zenodo_arg(parser)
    parser.add_argument('--out', default='figures/')
    args = parser.parse_args()
    check_zenodo_arg(parser, args)
    os.makedirs(args.out, exist_ok=True)

    setup_plotting_config()

    dann = load_summary(args.zenodo, 'hfd_TNG_ALL_Eagle100')
    mmd = load_summary(args.zenodo, 'hfm_TNG_ALL_Eagle100')
    standard = load_standard_method(args.zenodo, 'TNG_ALL', 'Eagle100')
    sm_test = dann['logMstar_true']

    sm_bins = np.arange(10, 12.1, 0.25)
    sm_bin_centers = 0.5 * (sm_bins[1:] + sm_bins[:-1])

    prec_dann, beta_dann = precision_beta(dann)
    prec_mmd, beta_mmd = precision_beta(mmd)
    prec_std, beta_std = precision_beta(standard)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=150, sharex=True)

    # left: precision
    ax = axes[0]
    ax.plot(sm_bin_centers, binned_median(sm_test, prec_std, sm_bins), label='Standard Method', lw=2, color='gray',
            ls='--', marker='^')
    ax.plot(sm_bin_centers, binned_median(sm_test, prec_dann, sm_bins), label='HaloFlow+DANN', lw=2, color=BLUE,
            marker='o')
    ax.plot(sm_bin_centers, binned_median(sm_test, prec_mmd, sm_bins), label='HaloFlow+MMD', lw=2, color=GREEN,
            marker='s')
    ax.set_ylabel(r'$\sigma_{\log M_\mathrm{h}}$ (dex)', fontsize=20)
    ax.set_xlabel(r'$\log M_*$', fontsize=20)
    ax.text(0.95, 0.05, r'\textsc{TNG} $\Rightarrow$ \textsc{Eagle}', transform=ax.transAxes, fontsize=16, ha='right')
    ax.set_ylim(0, 0.4)

    # right: beta
    ax = axes[1]
    ax.plot(sm_bin_centers, binned_median(sm_test, beta_std, sm_bins), label='_Standard_Method', lw=2, color='gray',
            marker='^')
    ax.plot(sm_bin_centers, binned_median(sm_test, beta_dann, sm_bins), label='_HaloFlow+DANN', lw=2, color=BLUE,
            marker='o')
    ax.plot(sm_bin_centers, binned_median(sm_test, beta_mmd, sm_bins), label='_HaloFlow+MMD', lw=2, color=GREEN,
            marker='s')

    ax.axhline(np.nanmedian(beta_dann), color=BLUE, ls='--', lw=2)
    ax.axhline(np.nanmedian(beta_mmd), color=GREEN, ls='--', lw=2)
    ax.axhline(np.nanmedian(beta_std), color='gray', ls='--', lw=2)
    ax.axhline(np.nan, color='k', lw=2, ls='--', label=r'Median $\beta$')
    ax.axhline(0.67, color='k', lw=3, label='Perfect \nCalibration')

    ax.set_xlabel(r'$\log M_*$', fontsize=20)
    ax.set_ylabel(r'$\beta$ (dex)', fontsize=20)   # label as in the published figure
    ax.set_ylim(0, 1.6)
    ax.legend(fontsize=16, loc='lower right', frameon=False)
    ax.set_xlim(10, 11.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=16, ncol=3, bbox_to_anchor=(0.5, 1.03))

    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'precision_bias.pdf'), dpi=150, bbox_inches='tight')

    print('median beta (all EAGLE test galaxies): DANN %.2f, MMD %.2f, standard method %.2f'
          % (np.nanmedian(beta_dann), np.nanmedian(beta_mmd), np.nanmedian(beta_std)))

'''
Fig. 8 of HaloFlow II: DRP (TARP) expected coverage of the joint (M*, Mh) posteriors
for all train=>test pairs (HaloFlow, HaloFlow+DANN, HaloFlow+MMD), from coverage/ of
the Zenodo archive.

    python fig8_drp_coverage.py --zenodo <path to the Zenodo archive> [--out figures/]

The curves are the ones in the paper (produced by drp_coverage_run.py, seed 0).
'''
import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from haloflow.config import setup_plotting_config
from zenodo_io import sims, load_coverage, add_zenodo_arg, check_zenodo_arg

title_names = {'TNG_ALL': 'TNG', 'Eagle100': 'Eagle', 'Simba100': 'Simba'}
methods = [('hf', 'HaloFlow', '#D55E00', '-'),
           ('hfd', 'HaloFlow+DANN', '#0072B2', '--'),
           ('hfm', 'HaloFlow+MMD', '#009E73', ':')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_zenodo_arg(parser)
    parser.add_argument('--out', default='figures/')
    args = parser.parse_args()
    check_zenodo_arg(parser, args)
    os.makedirs(args.out, exist_ok=True)

    setup_plotting_config()

    fig, ax = plt.subplots(3, 3, figsize=(14, 14), constrained_layout=True)
    for i, train_sim in enumerate(sims):        # columns: training simulation
        for j, test_sim in enumerate(sims):     # rows: test simulation
            sub = ax[j, i]
            sub.plot([0, 1], [0, 1], ls='--', color='k', lw=2)
            for m, label, color, ls in methods:
                if m != 'hf' and train_sim == test_sim:
                    continue
                alpha, ecp = load_coverage(args.zenodo, f'{m}_{train_sim}_{test_sim}')
                if i == j:
                    sub.plot(alpha, ecp, color='gray', ls=ls, lw=3.5, label='_nolegend_')
                else:
                    sub.plot(alpha, ecp, color=color, ls=ls, lw=3.5, label=label)

            sub.text(0.05, 0.95, r'$\textsc{%s} \Rightarrow \textsc{%s}$' % (title_names[train_sim], title_names[test_sim]),
                     transform=sub.transAxes, ha='left', va='top', fontsize=20)
            sub.set_xlim(0, 1)
            sub.set_ylim(0, 1)
            if j == 2:
                sub.set_xlabel('Credibility Level', fontsize=20)
            else:
                sub.set_xticklabels([])
            if i == 0:
                sub.set_ylabel('Expected Coverage', fontsize=20)
            else:
                sub.set_yticklabels([])

    handles, labels = ax[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=22, loc='upper center', bbox_to_anchor=(0.5, 1.04), frameon=False, ncol=3)

    fig.savefig(os.path.join(args.out, 'drp_coverage_all_methods.pdf'), bbox_inches='tight')

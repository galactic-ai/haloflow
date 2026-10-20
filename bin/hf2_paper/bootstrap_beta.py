'''
Bootstrap uncertainties (68% intervals) on the median beta and on the fraction
of test galaxies with beta < 1 for halo mass (Tables 2 and 3; HaloFlow II Sec. 4.2),
from the Zenodo posterior samples.

    python bootstrap_beta.py --zenodo <path to the Zenodo archive> [--out bootstrap_beta.txt]

Test rows (each line of sight treated as a galaxy) are resampled with replacement,
B = 1000, seed 0. The output is identical to the table in bootstrap/bootstrap_beta.txt
of the archive; the note at the end of that file (resampling whole galaxies, matched
by SubfindID) needs the feature catalogs and is not reproduced here.
'''
import os
import argparse

import numpy as np

from zenodo_io import sims, methods, load_run, add_zenodo_arg, check_zenodo_arg

names = {'TNG_ALL': 'TNG', 'Eagle100': 'EAGLE', 'Simba100': 'SIMBA'}
labels = {'hf': 'HaloFlow', 'hfd': '+DANN', 'hfm': '+MMD'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_zenodo_arg(parser)
    parser.add_argument('--out', default='bootstrap_beta.txt')
    args = parser.parse_args()
    check_zenodo_arg(parser, args)

    rng = np.random.default_rng(0)
    B = 1000

    rows = []
    for train in sims:
        for test in sims:
            for m in methods:
                if m != 'hf' and train == test:
                    continue
                y_true, y_nde = load_run(args.zenodo, f'{m}_{train}_{test}')
                y, t = y_nde[..., 1], y_true[:, 1]
                q16, q50, q84 = np.quantile(y, (.16, .5, .84), axis=1)
                b = np.abs(t - q50) / ((q84 - q16) / 2)

                N = len(b)
                idx = rng.integers(0, N, size=(B, N))
                med = np.median(b[idx], axis=1)
                frac = np.mean(b[idx] < 1, axis=1)
                rows.append((names[train], names[test], labels[m], N, np.median(b), *np.quantile(med, [.16, .84]),
                             np.mean(b < 1), *np.quantile(frac, [.16, .84])))

    with open(args.out, 'w') as f:
        f.write('train=>test        method      N     median beta [68% CI]      frac<1sigma [68% CI]\n')
        for r in rows:
            f.write(f'{r[0]+" => "+r[1]:18s} {r[2]:10s} {r[3]:5d}  {r[4]:.2f} [{r[5]:.2f}, {r[6]:.2f}]  '
                    f'(±{(r[6]-r[5])/2:.2f})   {r[7]:.2f} [{r[8]:.2f}, {r[9]:.2f}]\n')
    print(open(args.out).read())

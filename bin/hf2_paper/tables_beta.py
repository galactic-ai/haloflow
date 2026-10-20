'''
Tables 2, 3, 7 and 8 of HaloFlow II (median beta and fraction of test galaxies
with beta < 1, for halo and stellar mass) from the Zenodo posterior samples.

    python tables_beta.py --zenodo <path to the Zenodo archive>

beta = |true - p50| / ((p84 - p16)/2) per test galaxy, with np.quantile over the
1000 draws in posteriors/samples/{run}.txt.gz (zero placeholders included, as in
the paper). The values are checked against tables/*.csv of the archive.
'''
import os
import argparse

import numpy as np

from zenodo_io import sims, methods, load_run, add_zenodo_arg, check_zenodo_arg

names = {'TNG_ALL': 'TNG', 'Eagle100': 'EAGLE', 'Simba100': 'SIMBA'}


def read_csv(fname):
    out = {}
    for line in open(fname):
        if line.startswith('#') or line.startswith('run,'):
            continue
        cols = line.strip().split(',')
        out[cols[0]] = (float(cols[5]), float(cols[6]))
    return out


def print_table(title, vals, col):
    ''' layout of the paper: rows = test, columns = train, HaloFlow / +DANN / +MMD '''
    print(f'\n{title}')
    print(f'{"test / train":14s}' + ''.join(f'{names[s]:>22s}' for s in sims))
    for test in sims:
        row = f'{names[test]:14s}'
        for train in sims:
            v = [vals[f'{m}_{train}_{test}'][col] if f'{m}_{train}_{test}' in vals else None for m in methods]
            row += f'{" / ".join("-" if x is None else f"{x:.2f}" for x in v):>22s}'
        print(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_zenodo_arg(parser)
    args = parser.parse_args()
    check_zenodo_arg(parser, args)

    results = {'halo': {}, 'stellar': {}}
    for train in sims:
        for test in sims:
            for m in methods:
                if m != 'hf' and train == test:
                    continue
                run = f'{m}_{train}_{test}'
                y_true, y_nde = load_run(args.zenodo, run)
                q16, q50, q84 = np.quantile(y_nde, (0.16, 0.5, 0.84), axis=1)
                beta = np.abs(q50 - y_true) / (0.5 * (q84 - q16))
                for i, mass in enumerate(['stellar', 'halo']):
                    results[mass][run] = (np.median(beta[:, i]), np.mean(beta[:, i] < 1))

    print_table('Table 2: median beta, log Mh', results['halo'], 0)
    print_table('Table 3: fraction with beta < 1, log Mh', results['halo'], 1)
    print_table('Table 7: median beta, log M*', results['stellar'], 0)
    print_table('Table 8: fraction with beta < 1, log M*', results['stellar'], 1)

    # check against the machine-readable tables in the archive
    n_bad = 0
    for mass, fname in [('halo', 'table2_3_halo_mass.csv'), ('stellar', 'table7_8_stellar_mass.csv')]:
        ref = read_csv(os.path.join(args.zenodo, 'tables', fname))
        assert set(ref) == set(results[mass])
        for run, (med, frac) in results[mass].items():
            if abs(med - ref[run][0]) > 5e-5 or abs(frac - ref[run][1]) > 5e-5:
                print(f'MISMATCH {mass} {run}: {med:.4f} {frac:.4f} vs {ref[run]}')
                n_bad += 1
    print(f'\n{4 * len(results["halo"])} values checked against tables/*.csv: {n_bad} mismatches')
    raise SystemExit(n_bad > 0)

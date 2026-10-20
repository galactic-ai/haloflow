'''
DRP (TARP) expected coverage of the joint (M*, Mh) posteriors for all 21 train=>test
runs of HaloFlow II (Fig. 4 right, Fig. 8; coverage/ of the Zenodo archive).

    python drp_coverage_run.py [--out drp_samples/]

Needs the feature catalogs and the trained models (run from a subdirectory of bin/,
see haloflow.config.get_dat_dir). For each run, validate_npe draws 1000 unweighted
joint samples per test galaxy (no prior correction, no truncation; torch and numpy
seeded with 0), and tarp.get_tarp_coverage is evaluated with the settings used in
validate_npe (references='random', metric='euclidean', norm=False) plus seed=0;
norm=True is stored as a check. Output: {run}.npz with y_true, y_nde, and
alpha/ecp for norm=False (alpha_fig4, ecp_fig4), norm=True and the unseeded
validate_npe call.
'''
import os
import time
import argparse

import numpy as np
import torch
from tarp import get_tarp_coverage

from haloflow import config as C
from haloflow import data as D
from haloflow.npe import valid as V

obs = 'mags_morph_extra'
sims = ['TNG_ALL', 'Eagle100', 'Simba100']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='drp_samples/')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data_dir = C.get_dat_dir() + 'hf2/npe/'

    runs = []
    for train in sims:
        for test in sims:
            runs.append(('hf', train, test, dict()))
            if train != test:
                runs.append(('hfd', train, test, dict(with_dann=True)))
                runs.append(('hfm', train, test, dict(with_mmd=True)))

    for pre, train, test, kw in runs:
        key = f'{pre}_{train}_{test}'
        fname = os.path.join(args.out, key + '.npz')
        if os.path.exists(fname):
            print('skip', key, flush=True)
            continue

        t0 = time.time()
        torch.manual_seed(0)
        np.random.seed(0)
        _, alpha0, ecp0, y_nde = V.validate_npe(obs, test, train, obs, test, 'cpu', data_dir=data_dir, **kw)
        y_true, _ = D.hf2_centrals('test', obs, sim=test, version=1)
        assert y_nde.shape[0] == y_true.shape[0]

        S = np.swapaxes(y_nde, 0, 1)
        ecp_a, alpha_a = get_tarp_coverage(S, y_true, references='random', metric='euclidean', norm=False, seed=0)
        ecp_b, alpha_b = get_tarp_coverage(S, y_true, references='random', metric='euclidean', norm=True, seed=0)
        np.savez_compressed(fname, y_true=y_true.astype(np.float32), y_nde=y_nde.astype(np.float32),
                            alpha_fig4=alpha_a, ecp_fig4=ecp_a, alpha_norm=alpha_b, ecp_norm=ecp_b,
                            alpha_unseeded=alpha0, ecp_unseeded=ecp0)
        print(f'{key}: N={len(y_true)} {time.time() - t0:.0f}s', flush=True)

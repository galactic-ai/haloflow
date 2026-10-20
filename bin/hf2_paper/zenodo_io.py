'''
Readers for the HaloFlow II Zenodo archive, shared by the scripts in this folder.
Only numpy is needed (no torch, sbi or feature catalogs).
'''
import os

import numpy as np

sims = ['TNG_ALL', 'Eagle100', 'Simba100']
methods = ['hf', 'hfd', 'hfm']   # HaloFlow, HaloFlow+DANN, HaloFlow+MMD

# number of galaxies (x lines of sight) in hf2_centrals('all', 'mags_morph_extra', sim, version=1),
# i.e. train + test; needed to replay the train/test shuffle in sample_indices
n_all = {'TNG_ALL': 17201, 'Eagle100': 8888, 'Simba100': 8831}


def add_zenodo_arg(parser):
    parser.add_argument('--zenodo', default=os.environ.get('HALOFLOW_ZENODO'),
                        help='path to the Zenodo archive (default: $HALOFLOW_ZENODO)')


def check_zenodo_arg(parser, args):
    if args.zenodo is None:
        parser.error('give --zenodo or set HALOFLOW_ZENODO')


def load_summary(zdir, run):
    ''' per-galaxy summary of one run, as a dict of columns (see the archive README) '''
    fname = os.path.join(zdir, 'posteriors', 'summary', f'{run}_summary.txt')
    cols = ['galaxy_index', 'logMstar_true', 'logMh_true', 'Mstar_p16', 'Mstar_p50', 'Mstar_p84',
            'Mh_p16', 'Mh_p50', 'Mh_p84', 'beta_Mstar', 'beta_Mh']
    dat = np.loadtxt(fname)
    return {c: dat[:, i] for i, c in enumerate(cols)}


def load_standard_method(zdir, train='TNG_ALL', test='Eagle100'):
    ''' per-galaxy Mh summary of the standard-method (SHMR) baseline '''
    fname = os.path.join(zdir, 'posteriors', 'standard_method', f'standard_method_{train}_{test}_summary.txt')
    cols = ['galaxy_index', 'logMstar_true', 'logMh_true', 'Mh_p16', 'Mh_p50', 'Mh_p84', 'beta_Mh']
    dat = np.loadtxt(fname)
    return {c: dat[:, i] for i, c in enumerate(cols)}


def load_run(zdir, run):
    ''' true masses (N, 2) and posterior samples (N, 1000, 2) of one run; columns (log M*, log Mh) '''
    summ = load_summary(zdir, run)
    y_true = np.stack([summ['logMstar_true'], summ['logMh_true']], axis=1)

    samp = np.loadtxt(os.path.join(zdir, 'posteriors', 'samples', f'{run}.txt.gz'))
    n_test = len(y_true)
    n_draw = len(samp) // n_test
    # rows are ordered by galaxy, then draw
    assert np.array_equal(samp[:, 0], np.repeat(np.arange(n_test), n_draw))
    assert np.array_equal(samp[:, 1], np.tile(np.arange(n_draw), n_test))
    y_nde = samp[:, 2:].reshape(n_test, n_draw, 2)
    return y_true, y_nde


def load_coverage(zdir, run):
    ''' DRP (TARP) coverage curve: credibility level, expected coverage '''
    alpha, ecp = np.loadtxt(os.path.join(zdir, 'coverage', f'{run}_drp.txt'), unpack=True)
    return alpha, ecp


def sample_indices(logMh_true, sim, max_per_bin):
    ''' test galaxies shown in Figs. 4 and 5: identical to
    haloflow.npe.plotting.get_sample_indices('test', 'mags_morph_extra', sim, max_per_bin).

    get_sample_indices seeds numpy with 42 and then calls hf2_centrals, which seeds again
    and shuffles all n_all[sim] rows for the train/test split before the subset is drawn.
    That shuffle is replayed here so the global random state matches.
    '''
    np.random.seed(42)
    np.random.shuffle(np.arange(n_all[sim]))

    hm_bins = np.linspace(11, 15., 10)
    bin_indices = np.digitize(logMh_true, hm_bins)

    selected = []
    for i in range(1, len(hm_bins)):
        in_bin = np.where(bin_indices == i)[0]
        if len(in_bin) > max_per_bin:
            selected.extend(np.random.choice(in_bin, max_per_bin, replace=False))
        else:
            selected.extend(in_bin.tolist())
    return np.array(selected)

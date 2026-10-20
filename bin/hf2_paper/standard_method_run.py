'''
Standard-method (SHMR) baseline of HaloFlow II for TNG => EAGLE (Figs. 4 and 6;
posteriors/standard_method/ of the Zenodo archive).

    python standard_method_run.py [--out standard_method_TNG_ALL_Eagle100.npz]

Needs the feature catalogs and the photometry-only NPE h2.v1.TNG_ALL.mags (run from a
subdirectory of bin/, see haloflow.config.get_dat_dir). Inference only:
haloflow.npe.valid.standard_method with torch and numpy seeded with 0. Output: y_true
(EAGLE test set, log M*, log Mh) and hm_nde (1000 Mh draws per galaxy).
'''
import argparse

import numpy as np
import torch

from haloflow.npe import valid as V


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='standard_method_TNG_ALL_Eagle100.npz')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    y_true, _, hm_nde = V.standard_method(train_sim='TNG_ALL', test_sim='Eagle100')
    np.savez_compressed(args.out, y_true=y_true, hm_nde=hm_nde)

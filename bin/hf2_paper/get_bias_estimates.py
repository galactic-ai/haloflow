'''
Posterior samples behind Tables 2, 3, 7 and 8 of HaloFlow II, for all 21 train=>test
runs (HaloFlow, HaloFlow+DANN, HaloFlow+MMD).

    python get_bias_estimates.py [--out bias_estimate_results_new_weights.pkl]

Needs the feature catalogs and the trained models (run from a subdirectory of bin/,
see haloflow.config.get_dat_dir). For each run: 1000 draws from the 5-model NPE
ensemble (validate_npe), truncate_outliers, then the implicit-prior correction
weight_nde of the training simulation. The pickle written here was converted to
the text files in posteriors/ of the Zenodo archive; the paper's tables and
figures can be remade from those files with the other scripts in this folder.
'''
# %%
import argparse

import numpy as np

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--out', default='bias_estimate_results_new_weights.pkl')
args = parser.parse_args()

# %%
from haloflow import config as C
from haloflow.npe import valid as V
from haloflow import data as D
from haloflow import corr as Corr

from haloflow.util import bias_estimate, truncate_outliers

dat_dir = C.get_dat_dir()
data_dir = dat_dir + 'hf2/npe/'


# %%
sims = ['TNG_ALL', 'Simba100', 'Eagle100']
obs = 'mags_morph_extra'

# %%
bias_values = {}
results = {}

# %%
for i, train_sim in enumerate(sims):
    for j, dann_sim in enumerate(sims):
        # vanilla haloflow
        Y_test, _ = D.hf2_centrals('test', obs, dann_sim)
        # get number of test galaxies
        # number_test = len(Y_test)

        # np.random.seed(42)
        # idx = np.random.choice(len(Y_test), number_test, replace=False)
        # y_true = Y_test[idx]
        y_true = Y_test
        number_test = None

        print(f'{train_sim} -> {dann_sim}')

        _, _, _, y_nde_hf_unweighted = V.validate_npe(
            npe_train_obs=obs,
            dann_sim=dann_sim,
            npe_train_sim=train_sim,
            test_obs=obs,
            test_sim=dann_sim,
            device='cpu',
            data_dir=data_dir,
            n_ensemble=5,
            n_samples=1000,
            train_samples=number_test,
            version=1,
            with_dann=False,
            with_mmd=False,
        )
        y_nde_hf = truncate_outliers(y_nde_hf_unweighted) # 99th percentile

        y_nde_hf = Corr.weight_nde(y_nde_hf, train_sim)
        y_nde_hf_q0, y_nde_hf_q1, y_nde_hf_q2 = np.quantile(y_nde_hf, (0.16, 0.5, 0.84), axis=1)
        sigma_hf = (y_nde_hf_q2 - y_nde_hf_q0) / 2

        bias_estm_hf = bias_estimate(y_true, y_nde_hf_q1, sigma_hf)
        fraction_within_1sigma_hf = np.mean(bias_estm_hf[:, 1] < 1)

        key = f'hf_{train_sim}_{dann_sim}'
        bias_values[key] = bias_estm_hf
        results[key] = {
            'train_sim': train_sim,
            'test_sim': dann_sim,
            'method': 'haloflow',
            'y_true': y_true,
            'y_nde': y_nde_hf,
            'bias': bias_estm_hf,
            'fraction_1sigma': fraction_within_1sigma_hf,
        }

        print(f'Haloflow {np.median(bias_estm_hf[:, 1]):.2f}')
        # get fraction of halos within 1 sigma which is bias < 1
        print(f'Fraction within 1 sigma Haloflow {fraction_within_1sigma_hf:.2f}')

        if train_sim != dann_sim:
            # DANN true (haloflow + DANN)
            _, _, _, y_nde_dann_unweighted = V.validate_npe(
            npe_train_obs=obs,
            dann_sim=dann_sim,
            npe_train_sim=train_sim,
            test_obs=obs,
            test_sim=dann_sim,
            device='cpu',
            data_dir=data_dir,
            n_ensemble=5,
            n_samples=1000,
            train_samples=number_test,
            version=1,
            with_dann=True,
            with_mmd=False,
            )
            y_nde_dann = truncate_outliers(y_nde_dann_unweighted) # 99th percentile
    
            y_nde_dann = Corr.weight_nde(y_nde_dann, train_sim)
            y_nde_dann_q0, y_nde_dann_q1, y_nde_dann_q2 = np.quantile(y_nde_dann, (0.16, 0.5, 0.84), axis=1)
            sigma_dann = (y_nde_dann_q2 - y_nde_dann_q0) / 2

            bias_estm_dann = bias_estimate(y_true, y_nde_dann_q1, sigma_dann)
            fraction_within_1sigma_dann = np.mean(bias_estm_dann[:, 1] < 1)

            key = f'hfd_{train_sim}_{dann_sim}'
            bias_values[key] = bias_estm_dann
            results[key] = {
                'train_sim': train_sim,
                'test_sim': dann_sim,
                'method': 'haloflow_dann',
                'y_true': y_true,
                'y_nde': y_nde_dann,
                'bias': bias_estm_dann,
                'fraction_1sigma': fraction_within_1sigma_dann,
            }

            print(f'Haloflow+DANN {np.median(bias_estm_dann[:, 1]):.2f}')
            # get fraction of halos within 1 sigma which is bias < 1
            print(f'Fraction within 1 sigma Haloflow+DANN {fraction_within_1sigma_dann:.2f}')

            # MMD true (haloflow + MMD)

            _, _, _, y_nde_mmd_unweighted = V.validate_npe(
                npe_train_obs=obs,
                dann_sim=dann_sim,
                npe_train_sim=train_sim,
                test_obs=obs,
                test_sim=dann_sim,
                device='cpu',
                data_dir=data_dir,
                n_ensemble=5,
                n_samples=1000,
                train_samples=number_test,
                version=1,
                with_dann=False,
                with_mmd=True,
            )
            y_nde_mmd = truncate_outliers(y_nde_mmd_unweighted) # 99th percentile

            y_nde_mmd = Corr.weight_nde(y_nde_mmd, train_sim)
            y_nde_mmd_q0, y_nde_mmd_q1, y_nde_mmd_q2 = np.quantile(y_nde_mmd, (0.16, 0.5, 0.84), axis=1)
            sigma_mmd = (y_nde_mmd_q2 - y_nde_mmd_q0) / 2

            bias_estm_mmd = bias_estimate(y_true, y_nde_mmd_q1, sigma_mmd)
            fraction_within_1sigma_mmd = np.mean(bias_estm_mmd[:, 1] < 1)

            key = f'hfm_{train_sim}_{dann_sim}'
            bias_values[key] = bias_estm_mmd
            results[key] = {
                'train_sim': train_sim,
                'test_sim': dann_sim,
                'method': 'haloflow_mmd',
                'y_true': y_true,
                'y_nde': y_nde_mmd,
                'bias': bias_estm_mmd,
                'fraction_1sigma': fraction_within_1sigma_mmd,
            }

            # get fraction of halos within 1 sigma which is bias < 1
            print(f'Haloflow+MMD {np.median(bias_estm_mmd[:, 1]):.2f}')
            print(f'Fraction within 1 sigma Haloflow+MMD {fraction_within_1sigma_mmd:.2f}')


# %%
# save bias values
# import pickle
# with open('bias_values_new.pkl', 'wb') as f:
#     pickle.dump(bias_values, f)


# save results
import pickle
with open(args.out, 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


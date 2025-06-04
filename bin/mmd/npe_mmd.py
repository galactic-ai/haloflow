import sys

from haloflow.config import get_dat_dir
from haloflow.npe.optuna_training import NPEOptunaTraining
from haloflow.mmd.get_preds import get_mmd_preds
import haloflow.data as D

import torch

##################################################################################
cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")
##################################################################################
obs = sys.argv[1]
sim = sys.argv[2]
dann_sim = sys.argv[3] if len(sys.argv) > 3 else 'Simba100'
all_sims = ['TNG_ALL', 'Eagle100', 'Simba100']

if sim not in all_sims: raise ValueError

FP = get_dat_dir() + f'hf2/mmd/models/mmd_model_v2_to_{dann_sim}_{obs}.pt'

##################################################################################
# read in training data 
y_test, x_test = D.hf2_centrals('test', obs, sim=sim, version=1)

_, x_test = get_mmd_preds(FP, obs, sim)

# Optuna Parameters
n_trials    = 1000
study_name  = f'h2.mmd.v2.m{dann_sim}.{sim}.{obs}'
print(f"Running Optuna for {study_name} with {n_trials} trials...")

output_dir = get_dat_dir() + 'hf2/npe/'

npe = NPEOptunaTraining(
        y_test, x_test, 
        n_trials, 
        study_name,
        output_dir,
        n_jobs=8,
        device=device
)
study = npe()

print("  Number of finished trials: %i" % len(study.trials))

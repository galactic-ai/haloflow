#/bin/python 
import sys

from haloflow.dann.get_preds import get_dann_preds
from haloflow.npe.optuna_training import NPEOptunaTraining
import haloflow.data as D

import torch

##################################################################################
cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")
##################################################################################
obs = sys.argv[1]
sim = sys.argv[2]
all_sims = ['TNG50', 'TNG100', 'Eagle100', 'Simba100']

if sim not in all_sims: raise ValueError

##################################################################################
# read in training data 
y_test, x_test = D.hf2_centrals('test', obs, sim=sim, version=1)
label_pred, domain_pred = get_dann_preds('../../data/hf2/dann/models/dann_model_TNG50_TNG100_Eagle100_to_Simba100_mags_lr0.001_bs32_e100_2025-03-05.pt', obs, sim)
x_test = label_pred.detach().numpy()

# Optuna Parameters
n_trials    = 1000
study_name  = 'h2.dann.v1.%s.%s' % (sim, obs) 

output_dir = '../../data/hf2/dann/npe/'

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

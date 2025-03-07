#/bin/python 
import sys

from haloflow.npe.optuna_training import NPEOptunaTraining
import haloflow.data as D

import torch


##################################################################################
cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")
##################################################################################
obs = sys.argv[1]
sim = sys.argv[2]

if sim not in ['TNG50', 'TNG100', 'TNG_ALL', 'Eagle100', 'Simba100']: raise ValueError

# read in training data 
y_train, x_train = D.hf2_centrals('train', obs, sim=sim, version=1) 

# Optuna Parameters
n_trials    = 1000
study_name  = 'h2.v1.%s.%s' % (sim, obs) 

output_dir = '/xdisk/chhahn/chhahn/haloflow/hf2/npe'

npe = NPEOptunaTraining(
        y_train, x_train, 
        n_trials, 
        study_name,
        output_dir,
        n_jobs=8,
        device=device
)
study = npe()

print("  Number of finished trials: %i" % len(study.trials))

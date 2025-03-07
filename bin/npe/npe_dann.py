#/bin/python 
import sys
from glob import glob

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

rem_sims = '_'.join([s for s in all_sims if s != 'Simba100'])
# fp = f'../../data/hf2/dann/models/dann_model_{rem_sims}_to_{sim}_{obs}_*.pt'
fp = f'../../data/hf2/dann/models/dann_model_{rem_sims}_to_Simba100_{obs}_*.pt'
fp = glob(fp)[0]

##################################################################################
# read in training data 
y_test, x_test = D.hf2_centrals('test', obs, sim=sim, version=1)
label_pred, domain_pred = get_dann_preds(fp, obs, sim)
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

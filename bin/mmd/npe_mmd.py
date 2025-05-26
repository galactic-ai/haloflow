import sys
from glob import glob

from matplotlib.pyplot import sca

from haloflow.config import get_dat_dir
from haloflow.util import get_all_data_from_loader
from haloflow.dann.data_loader import SimulationDataset
from haloflow.npe.optuna_training import NPEOptunaTraining
from haloflow.mmd.models import MMDModel
from haloflow.mmd.get_preds import get_mmd_preds
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

FP = get_dat_dir() + f'hf2/mmd/models/mmd_best_model_to_Simba100_{obs}.pth'

##################################################################################
# read in training data 
y_test, x_test = D.hf2_centrals('test', obs, sim=sim, version=1)

_, x_test = get_mmd_preds(FP, obs, sim)

# Optuna Parameters
n_trials    = 1000
study_name  = 'h2.mmd.v1.%s.%s' % (sim, obs) 

output_dir = '../../data/hf2/npe/'

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

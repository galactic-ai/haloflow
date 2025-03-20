#/bin/python 
import sys

from haloflow.dann.model import DANNModel
from haloflow.dann.evalutate import evaluate
from haloflow.npe.optuna_training import NPEOptunaTraining
from haloflow.config import get_dat_dir
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

MODEL_NAME = f'dann_model_to_{sim}_{obs}'
FP = get_dat_dir() + f'hf2/dann/models/{MODEL_NAME}.pt'

##################################################################################
# read in training data 
y_test, x_test = D.hf2_centrals('test', obs, sim=sim, version=1)

input_dim = x_test.shape[1]
model = DANNModel(input_dim=input_dim).to(device)
model.load_state_dict(torch.load(FP, map_location=device))

y_test, label_pred, _ = evaluate(model, obs, sim, device=device)

# Optuna Parameters
n_trials    = 1000
study_name  = 'h2.dann.v1.%s.%s' % (sim, obs) 

output_dir = '../../data/hf2/dann/npe/'

npe = NPEOptunaTraining(
        y_test, label_pred, 
        n_trials, 
        study_name,
        output_dir,
        n_jobs=8,
        device=device
        )
study = npe()

print("Number of finished trials: %i" % len(study.trials))

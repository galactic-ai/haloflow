#/bin/python 
import sys

from haloflow.dann.model import DANNModel
from haloflow.dann.evalutate import evaluate
from haloflow.npe.optuna_training import NPEOptunaTraining
from haloflow.config import get_dat_dir
import haloflow.data as D

import torch
import numpy as np

##################################################################################
cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")
##################################################################################
obs = sys.argv[1]
sim = sys.argv[2]
dann_sim = sys.argv[3] if len(sys.argv) > 3 else 'Simba100'
all_sims = ['TNG_ALL', 'Eagle100', 'Simba100']

if sim not in all_sims: raise ValueError

MODEL_NAME = f'dann_model_v3_to_{dann_sim}_{obs}'
FP = get_dat_dir() + f'hf2/dann/models/{MODEL_NAME}.pt'
FP_mean_std = get_dat_dir() + f'hf2/dann/models/{MODEL_NAME}_mean_std.npz'

##################################################################################
# read in training data 
y_train, x_train = D.hf2_centrals('train', obs, sim=sim, version=1)

input_dim = x_train.shape[1]
model = DANNModel(input_dim=input_dim).to(device)
model.load_state_dict(torch.load(FP, map_location=device))

array = np.load(FP_mean_std)
mean_, std_ = array['mean'], array['std']

y_train, label_pred, loss, r2, cX = evaluate(model, obs, sim, device=device, dataset='train', mean_=mean_, std_=std_)
print(f"Loss: {loss:.4f}, R2: {r2:.4f}")

# Optuna Parameters
n_trials    = 1000
study_name  = f'h2.dann.v3.m{dann_sim}.{sim}.{obs}.feature.extract'


output_dir = get_dat_dir() + 'hf2/npe/'

npe = NPEOptunaTraining(
        y_train, 
        cX, 
        n_trials, 
        study_name,
        output_dir,
        n_jobs=8,
        device=device
        )
study = npe()

print("Number of finished trials: %i" % len(study.trials))

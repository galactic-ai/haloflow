#/bin/python 
import os,sys

import haloflow.data as D
from haloflow.dann import data_loader as DL
from haloflow.dann import model as M
from haloflow import config as C

import torch
from torch.utils.tensorboard.writer import SummaryWriter

import optuna 
#from sbi import neural_nets as nn
from sbi import inference as Inference
from sbi import utils as Ut

def get_all_data_from_loader(dataloader):
    all_X, all_Y = [], []
    for batch in dataloader:
        X_batch, Y_batch, _ = batch 
        all_X.append(X_batch)
        all_Y.append(Y_batch)
    
    all_X = torch.cat(all_X, dim=0)  
    all_Y = torch.cat(all_Y, dim=0)  
    return all_X, all_Y

##################################################################################
cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")
##################################################################################
all_sims = ['TNG50', 'TNG100', 'Eagle100', 'Simba100']
obs = sys.argv[1]
sim = sys.argv[2]

if sim not in all_sims: raise ValueError

####
dat_dir = C.get_dat_dir()
dataset = DL.SimulationDataset(
    sims=all_sims,
    obs=obs,
    data_dir=dat_dir)

_, test_loader = dataset.get_train_test_loaders(
    [s for s in all_sims if s != sim],
    sim,
    32
)
sample_X, _, _ = next(iter(test_loader))

all_X, all_Y = get_all_data_from_loader(test_loader)
####

# model
fp = '../../data/hf2/dann/models/dann_model_TNG50_TNG100_Eagle100_to_Simba100_mags_lr0.001_bs32_e100_2025-03-05.pt'
model_DANN = M.DANN(input_dim=sample_X.shape[1], 
           feature_layers=[128, 64, 32], 
           label_layers=[32, 16, 8], 
           domain_layers=[32, 16, 8],
           num_domains=3,
           alpha=0)
model_DANN.load_state_dict(torch.load(fp))
model_DANN.eval()

# predict
label_pred, domain_pred = model_DANN(all_X)
###


##################################################################################
# read in training data 
##################################################################################
y_test, x_test = D.hf2_centrals('test', obs, sim=sim, version=1) 
y_test = label_pred.detach().numpy()

##################################################################################
# prior 
##################################################################################
lower_bounds = torch.tensor([8., 8.]) # training set only includes galaxies with logMstar/Mun > 9
upper_bounds = torch.tensor([14., 15.])

prior = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=device)
##################################################################################
# OPTUNA
##################################################################################
# Optuna Parameters
n_trials    = 1000
study_name  = 'h2.dann.v1.%s.%s' % (sim, obs) 

output_dir = '../../data/hf2/dann/npe/'

n_jobs     = 8
if not os.path.isdir(os.path.join(output_dir, study_name)): 
    os.system('mkdir %s' % os.path.join(output_dir, study_name))
storage    = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
n_startup_trials = 20

n_blocks_min, n_blocks_max = 2, 5 
n_transf_min, n_transf_max = 2, 5 
n_hidden_min, n_hidden_max = 32, 128 
n_lr_min, n_lr_max = 5e-6, 1e-3 


def Objective(trial):
    ''' bojective function for optuna 
    '''
    # Generate the model                                         
    n_blocks = trial.suggest_int("n_blocks", n_blocks_min, n_blocks_max)
    n_transf = trial.suggest_int("n_transf", n_transf_min,  n_transf_max)
    n_hidden = trial.suggest_int("n_hidden", n_hidden_min, n_hidden_max, log=True)
    lr = trial.suggest_float("lr", n_lr_min, n_lr_max, log=True) 
    neural_posterior = Ut.posterior_nn('maf', 
            hidden_features=n_hidden, 
            num_transforms=n_transf, 
            num_blocks=n_blocks, 
            use_batch_norm=True)

    anpe = Inference.SNPE(prior=prior,
            density_estimator=neural_posterior,
            device=device, 
            summary_writer=SummaryWriter('%s/%s/%s.%i' % 
                (output_dir, study_name, study_name, trial.number)))

    anpe.append_simulations( 
            torch.tensor(y_test, dtype=torch.float32).to(device), 
            torch.tensor(x_test, dtype=torch.float32).to(device))

    p_theta_x_est = anpe.train(
            training_batch_size=50,
            learning_rate=lr, 
            show_train_summary=True)

    # save trained NPE  
    qphi    = anpe.build_posterior(p_theta_x_est)
    fqphi   = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    torch.save(qphi, fqphi)

    best_valid_log_prob = anpe._summary['best_validation_log_prob'][0]

    return -1*best_valid_log_prob

sampler     = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials) 
study       = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, directions=["minimize"], load_if_exists=True) 

study.optimize(Objective, n_trials=n_trials, n_jobs=n_jobs)
print("  Number of finished trials: %i" % len(study.trials))

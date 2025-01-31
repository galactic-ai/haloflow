#!/bin/python 
''' script to run validation tests on NPE  
'''
import sys

import torch
from haloflow.npe import valid as V
from haloflow.npe import plotting

##################################################################################
cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")
##################################################################################
obs = sys.argv[1]
sim = sys.argv[2]
version = 1

if sim not in ['TNG50', 'TNG100', 'TNG_ALL', 'Eagle100', 'Simba100']: raise ValueError
if obs not in ['mags', 'mags_morph']: raise ValueError

ranks, alpha, ecp, y_nde = V.validate_npe(train_obs=obs, train_sim=sim, 
                                   test_obs=obs, test_sim=sim, 
                                   version=version, device=device)

##################################################################################
# rank statistic plot 
##################################################################################
fig = plotting.plot_rank_statistics([ranks], labels=[sim])
fig.savefig(f'/xdisk/chhahn/chhahn/haloflow/hf2/npe/h2.v{version}.{sim}.{obs}.rank.png',
        bbox_inches='tight')

##################################################################################
# TARP coverage test  
##################################################################################
fig, ax = plotting.plot_coverage([alpha], [ecp], labels=[sim])
fig.savefig(f'/xdisk/chhahn/chhahn/haloflow/hf2/npe/h2.v{version}.{sim}.{obs}.tarp.png',
        bbox_inches='tight')

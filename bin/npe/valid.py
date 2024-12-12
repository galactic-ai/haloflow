#!/bin/python 
''' script to run validation tests on NPE  
'''
import os
import numpy as np

import torch
from haloflow import data as D
from haloflow import util as U

from tarp import get_drp_coverage

import corner as DFM
# --- plotting ---
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False

##################################################################################
cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")
##################################################################################
obs = sys.argv[1]
sim = sys.argv[2]

##################################################################################
# read ensemble of NPEs 
##################################################################################
qphis = U.read_best_ndes('h2.v1.%s.%s' % (sim, obs), 
        n_ensemble=5, device=device, 
        dat_dir='/scratch/gpfs/chhahn/haloflow/hf2/npe', # hardcoded to run on della ATM
        verbose=True)

##################################################################################
# read test data 
##################################################################################
Y_test, X_test = D.hf2_centrals('test', obs, sim=sim, version=1)

##################################################################################
# coverage tests 
##################################################################################
n_sample = 10000

ranks = []
y_nde = [] 
for i in trange(X_test.shape[0]): 
    y_samp = []
    for qphi in qphis: 
        _samp = qphi.sample((int(n_sample/len(qphis)),),
                               x=torch.tensor(X_test[i], dtype=torch.float32).to(device), 
                               show_progress_bars=False)
        y_samp.append(_samp.detach().cpu().numpy())
    y_nde.append(np.concatenate(np.array(y_samp), axis=0)) 
    
    _ranks = []
    for i_dim in range(y_nde[-1].shape[1]): 
        _ranks.append(np.mean(y_nde[-1][:,i_dim].flatten() < Y_test[i, i_dim]))
    ranks.append(_ranks)
    
ranks = np.array(ranks)
y_nde = np.array(y_nde)


# calculate TARP coverages 
alpha, ecp = get_drp_coverage(np.swapaxes(y_nde, 0, 1), 
        Y_test, references="random", metric="euclidean")

##################################################################################
# rank statistic plot 
##################################################################################
fig = plt.figure(figsize=(12,3.5))
sub = fig.add_subplot(121)

_ = sub.hist(ranks[:,0], range=(0., 1), bins=20, histtype='step', density=True, linewidth=2)

sub.plot([0., 1.], [1., 1.], c='k', ls='--')
sub.text(0.05, 0.95, r'$\log M_*$', fontsize=20, transform=sub.transAxes, ha='left', va='top')
sub.set_xlabel('rank statistics', fontsize=20)
sub.set_xlim(0., 1.)
sub.set_ylim(0., 3.)
sub.set_yticks([])

sub = fig.add_subplot(122)

_ = sub.hist(ranks[:,1], range=(0., 1), bins=20, histtype='step', density=True, linewidth=2)

sub.plot([0., 1.], [1., 1.], c='k', ls='--')
sub.text(0.05, 0.95, r'$\log M_h$', fontsize=20, transform=sub.transAxes, ha='left', va='top')
sub.set_xlabel('rank statistics', fontsize=20)
sub.set_xlim(0., 1.)
sub.set_ylim(0., 3.)
sub.set_yticks([])
fig.savefig('/scratch/gpfs/chhahn/haloflow/hf2/npe/h2.v1.%s.%s.rank.png' % (sim, obs), 
        bbox_inches='tight')

##################################################################################
# TARP coverage test  
##################################################################################
fig, ax = plt.subplots(1, 1, figsize=(6,6))
ax.plot([0, 1], [0, 1], ls="--", color="k")
ax.plot(alpha, ecp, c='C0')
ax.legend(loc='lower right', fontsize=15)
ax.set_ylabel("Expected Coverage", fontsize=20)
ax.set_ylim(0., 1.)
ax.set_xlabel("Credibility Level", fontsize=20)
ax.set_xlim(0., 1.)
fig.savefig('/scratch/gpfs/chhahn/haloflow/hf2/npe/h2.v1.%s.%s.tarp.pdf' % (sim, obs), 
        bbox_inches='tight')

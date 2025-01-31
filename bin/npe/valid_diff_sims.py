import concurrent.futures
import torch

from haloflow.npe import valid as V
from haloflow.npe import plotting

cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")

sims = ['Simba100', 'Eagle100', 'TNG50', 'TNG100', 'TNG_ALL'] 
obs = 'mags'

# Process the simulations in parallel to save time
# won't make a figure on same train and test simulation
def process_simulations_figures(train_sim):
    print(f'Processing {train_sim}')
    all_ranks = []
    all_alpha = []
    all_ecp = []
    for test_sim in sims:
        if train_sim == test_sim:
            continue
        
        ranks, alpha, ecp, y_nde = V.validate_npe(train_obs=obs, train_sim=train_sim, test_obs=obs, test_sim=test_sim, version=1)
        
        all_ranks.append(ranks)
        all_alpha.append(alpha)
        all_ecp.append(ecp)

    fig = plotting.plot_rank_statistics(all_ranks, labels=[f'{train_sim}_{test_sim}' for test_sim in sims if train_sim != test_sim])
    fig.savefig('/xdisk/chhahn/chhahn/haloflow/hf2/npe/h2.v1.%s.%s.rank_crossvalid.png' % (train_sim, obs), 
        bbox_inches='tight')
    
    fig.clf()

    fig, ax = plotting.plot_coverage(all_alpha, all_ecp, labels=[f'{train_sim}_{test_sim}' for test_sim in sims if train_sim != test_sim])
    fig.savefig('/xdisk/chhahn/chhahn/haloflow/hf2/npe/h2.v1.%s.%s.tarp_crossvalid.png' % (train_sim, obs), 
        bbox_inches='tight')
    
    fig.clf()

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(process_simulations_figures, sims)

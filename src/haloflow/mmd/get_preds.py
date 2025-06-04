import torch
import numpy as np

from .. import config as C
from .. import data as D
from ..mmd import model as M

DAT_DIR = C.get_dat_dir()
ALL_SIMS = ["TNG_ALL", "Eagle100", "Simba100"]


def get_mmd_preds(fp, obs, sim):
    y_test, x_test = D.hf2_centrals('test', obs, sim=sim, version=1)

    if sim not in ALL_SIMS:
        raise ValueError(f"Simulation {sim} not recognized. Choose from {ALL_SIMS}.")

    # Load global mean and std normalization parameters
    global_stats = np.load(DAT_DIR + 'hf2/mmd/models/global_mean_std.npz', allow_pickle=True)
    g_mean = global_stats['mean']
    g_std = global_stats['std']

    x_test = (x_test - g_mean) / g_std
    
    all_X = torch.tensor(x_test, dtype=torch.float32)
    input_dim = all_X.shape[1]

    
    model = M.MMDModel(input_dim, output_dim=2)
    model.load_state_dict(torch.load(fp))
    
    model.eval()
    
    with torch.no_grad():
        features, label_pred = model(all_X)
    
    features = features.detach().numpy()
    label_pred = label_pred.detach().numpy()
    
    return features, label_pred
    
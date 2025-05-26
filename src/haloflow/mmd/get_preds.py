import torch

from haloflow import config as C
from haloflow.dann import data_loader as DL
from haloflow.mmd import models as M
from haloflow.util import get_all_data_from_loader

DAT_DIR = C.get_dat_dir()
ALL_SIMS = ["TNG50", "TNG100", "Eagle100", "Simba100"]


def get_mmd_preds(fp, obs, sim):
    dataset = DL.SimulationDataset(sims=ALL_SIMS, obs=obs, data_dir=DAT_DIR)
    train_sims = [s for s in ALL_SIMS if s != sim]
    
    _, test_loader = dataset.get_train_test_loaders(
        train_sims, sim, 32
    )
    sample_X, _, _ = next(iter(test_loader))
    input_dim = sample_X.shape[1]

    all_X, _ = get_all_data_from_loader(test_loader)
    # scaler = dataset.scaler_Y
    
    model = M.MMDModel(input_dim, output_dim=2)
    model.load_state_dict(torch.load(fp))
    
    model.eval()
    
    with torch.no_grad():
        features, label_pred = model(all_X)
    
    features = features.detach().numpy()
    label_pred = label_pred.detach().numpy()
    
    # label_pred = scaler.inverse_transform(label_pred)
    
    return features, label_pred
    
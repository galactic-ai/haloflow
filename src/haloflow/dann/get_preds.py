import torch

from haloflow import config as C
from haloflow.dann import data_loader as DL
from haloflow.dann import model as M

dat_dir = C.get_dat_dir()
all_sims = ['TNG50', 'TNG100', 'Eagle100', 'Simba100']

def get_all_data_from_loader(dataloader):
    all_X, all_Y = [], []
    for batch in dataloader:
        X_batch, Y_batch, _ = batch 
        all_X.append(X_batch)
        all_Y.append(Y_batch)
    
    all_X = torch.cat(all_X, dim=0)  
    all_Y = torch.cat(all_Y, dim=0)  
    return all_X, all_Y


def get_dann_preds(fp, obs, sim):
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

    all_X, _ = get_all_data_from_loader(test_loader)
    ####

    # model
    model_DANN = M.DANN(input_dim=sample_X.shape[1], 
            feature_layers=[128, 64, 32], 
            label_layers=[32, 16, 8], 
            domain_layers=[32, 16, 8],
            num_domains=3,
            alpha=0)
    model_DANN.load_state_dict(torch.load(fp))
    model_DANN.eval()

    # predict
    with torch.no_grad():
        label_pred, domain_pred = model_DANN(all_X)
    
    return label_pred, domain_pred
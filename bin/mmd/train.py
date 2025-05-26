from haloflow.config import get_dat_dir
from haloflow.dann.data_loader import SimulationDataset
from haloflow.mmd.models import MMDModel
from haloflow.mmd.utils import mmd_loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

SIMS = ['TNG50', 'TNG100', 'Eagle100', 'Simba100']
OBS = 'mags'
TEST_SIM = 'Simba100'
TRAIN_SIMS = [sim for sim in SIMS if sim != TEST_SIM]

FP = get_dat_dir() + f'hf2/mmd/models/mmd_best_model_to_{TEST_SIM}_{OBS}.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_dim = None  # feature dimension of the catalog data
output_dim = 2  # number of classes
lambda_mmd = 0.5  # weight for MMD loss
sigma = 1.0  # kernel bandwidth
lr = 0.0001
epochs = 350

# Load data: separate loaders for each domain
sim_dataset = SimulationDataset(SIMS, OBS, get_dat_dir())
loader_A, _ = sim_dataset.get_train_test_loaders([TRAIN_SIMS[0]], TEST_SIM)
loader_B, _ = sim_dataset.get_train_test_loaders([TRAIN_SIMS[1]], TEST_SIM)
loader_C, _ = sim_dataset.get_train_test_loaders([TRAIN_SIMS[2]], TEST_SIM)

if input_dim is None:
    input_dim = next(iter(loader_A))[0].shape[1]

# Initialize models
mmd_model = MMDModel(input_dim, output_dim).to(device)
optimizer = optim.AdamW(mmd_model.parameters(), lr=lr, weight_decay=5e-4)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

criterion = nn.MSELoss()

# early stopping
patience = 10
best_loss = float('inf')

for epoch in range(epochs):
    total_loss = 0
    mmd_total = 0
    cls_total = 0

    for i, (batch_A, batch_B, batch_C) in enumerate(zip(loader_A, loader_B, loader_C)):
        x_A, y_A, _ = batch_A
        x_B, y_B, _ = batch_B
        x_C, y_C, _ = batch_C
        
        x_A = x_A.to(device)
        x_B = x_B.to(device)
        x_C = x_C.to(device)
        
        y_A = y_A.to(device)
        y_B = y_B.to(device)
        y_C = y_C.to(device)
        

        features_A, outputs_A = mmd_model(x_A)
        features_B, outputs_B = mmd_model(x_B)
        features_C, outputs_C = mmd_model(x_C)        

        # Compute MMD for all pairs
        mmd_AB = mmd_loss(features_A, features_B, sigma)
        mmd_AC = mmd_loss(features_A, features_C, sigma)
        mmd_BC = mmd_loss(features_B, features_C, sigma)
        total_mmd = mmd_AB + mmd_AC + mmd_BC

        # Compute classification loss
        cls_loss = (criterion(outputs_A, y_A) + 
                    criterion(outputs_B, y_B) + 
                    criterion(outputs_C, y_C)) / 3

        # Total loss
        loss = cls_loss + lambda_mmd * total_mmd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

        total_loss += loss.item()
        mmd_total += total_mmd.item()
        cls_total += cls_loss.item()

    avg_loss = total_loss / (i + 1)
    avg_mmd = mmd_total / (i + 1)
    avg_cls = cls_total / (i + 1)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, MMD Loss={avg_mmd:.4f}, CLS Loss={avg_cls:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience = 10  # reset patience
        torch.save(mmd_model.state_dict(), FP)
    else:
        patience -= 1
        if patience <= 0:
            print("Early stopping triggered.")
            break
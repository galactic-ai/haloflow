import argparse
import numpy as np
import torch
import torch.optim as optim

from haloflow.config import get_dat_dir
from haloflow.mmd.models import MMDModel
from haloflow.mmd.utils import mmd_loss
from haloflow.schechter import schechter_logmass
from haloflow.util import weighted_mse_loss
import haloflow.data as D

# set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)

# parse command line arguments
parser = argparse.ArgumentParser(description='Train a MMD model')
parser.add_argument('--obs', type=str, default='mags_morph_extra', help='Observation to use')
parser.add_argument('--sim', type=str, default='TNG_ALL', help='Simulation to use for testing (domain adaptation)')


args = parser.parse_args()
OBS = args.obs
TEST_SIM = args.sim


SIMS = ['TNG_ALL', 'Eagle100', 'Simba100']
TRAIN_SIMS = [sim for sim in SIMS if sim != TEST_SIM]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = f'mmd_model_v2_to_{TEST_SIM}_{OBS}'
FP = get_dat_dir() + f'hf2/mmd/models/{MODEL_NAME}.pt'

# Hyperparameters
input_dim = None  # feature dimension of the catalog data
output_dim = 2  # number of classes
lambda_mmd = 0.5  # weight for MMD loss
sigma = 1.0  # kernel bandwidth
lr = 0.0001
epochs = 350
batch_size = 512  # Add batch size

# Load data: separate loaders for each domain
y_t0, X_t0 = D.hf2_centrals("train", obs=OBS, sim=TRAIN_SIMS[0])
y_t1, X_t1 = D.hf2_centrals("train", obs=OBS, sim=TRAIN_SIMS[1])
y_test, X_test = D.hf2_centrals("train", obs=OBS, sim=TEST_SIM)

sche_weights_t0 = 1 / schechter_logmass(y_t0[:, 0])
sche_weights_t0 = sche_weights_t0 / np.min(sche_weights_t0)
sche_weights_t0 = np.clip(sche_weights_t0, 0, 1e2)

sche_weights_t1 = 1 / schechter_logmass(y_t1[:, 0])
sche_weights_t1 = sche_weights_t1 / np.min(sche_weights_t1)
sche_weights_t1 = np.clip(sche_weights_t1, 0, 1e2)

# Global mean, std normalization
all_data = np.concatenate((X_t0, X_t1, X_test))
g_mean = np.mean(all_data, axis=0)
g_std = np.std(all_data, axis=0)
X_t0 = (X_t0 - g_mean) / g_std
X_t1 = (X_t1 - g_mean) / g_std
X_test = (X_test - g_mean) / g_std

# save the global mean and std
np.savez(get_dat_dir() + 'hf2/mmd/models/global_mean_std.npz', mean=g_mean, std=g_std)

# Convert to tensors
X_t0_tensor = torch.tensor(X_t0, dtype=torch.float32).to(DEVICE)
y_t0_tensor = torch.tensor(y_t0, dtype=torch.float32).to(DEVICE)
weights_t0_tensor = torch.tensor(sche_weights_t0, dtype=torch.float32).to(DEVICE)
weights_t0_tensor = weights_t0_tensor.unsqueeze(1).expand(-1, 2)

X_t1_tensor = torch.tensor(X_t1, dtype=torch.float32).to(DEVICE)
y_t1_tensor = torch.tensor(y_t1, dtype=torch.float32).to(DEVICE)
weights_t1_tensor = torch.tensor(sche_weights_t1, dtype=torch.float32).to(DEVICE)
weights_t1_tensor = weights_t1_tensor.unsqueeze(1).expand(-1, 2)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)
weights_test_tensor = torch.ones_like(y_test_tensor, dtype=torch.float32).to(DEVICE)

loader_A = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_t0_tensor, y_t0_tensor, weights_t0_tensor),
    batch_size=batch_size, shuffle=True
)
loader_B = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_t1_tensor, y_t1_tensor, weights_t1_tensor),
    batch_size=batch_size, shuffle=True
)

loader_test = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor, weights_test_tensor),
    batch_size=batch_size, shuffle=False
)

if input_dim is None:
    input_dim = X_t0_tensor.shape[1]

# Initialize models
mmd_model = MMDModel(input_dim, output_dim).to(DEVICE)
optimizer = optim.AdamW(mmd_model.parameters(), lr=lr, weight_decay=5e-4)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

criterion = weighted_mse_loss

# early stopping
patience = 10
best_loss = float('inf')

for epoch in range(epochs):
    total_loss = 0
    mmd_total = 0
    cls_total = 0
    
    # Create iterators and get minimum length
    iter_A = iter(loader_A)
    iter_B = iter(loader_B)
    iter_test = iter(loader_test)
    min_batches = min(len(loader_A), len(loader_B), len(loader_test))

    for i in range(min_batches):
        try:
            batch_A = next(iter_A)
            batch_B = next(iter_B)
            batch_C = next(iter_test)
        except StopIteration:
            break
            
        x_A, y_A, weights_A = batch_A
        x_B, y_B, weights_B = batch_B
        x_C, _, _ = batch_C

        x_A = x_A.to(DEVICE)
        x_B = x_B.to(DEVICE)
        x_C = x_C.to(DEVICE)

        y_A = y_A.to(DEVICE)
        y_B = y_B.to(DEVICE)

        features_A, outputs_A = mmd_model(x_A)
        features_B, outputs_B = mmd_model(x_B)
        features_C, outputs_C = mmd_model(x_C)

        # Compute MMD for all pairs
        mmd_BC = mmd_loss(features_B, features_C, sigma=sigma)
        mmd_AC = mmd_loss(features_A, features_C, sigma=sigma)
        total_mmd = (mmd_BC + mmd_AC)

        # Compute classification loss
        cls_loss = (criterion(outputs_A, y_A, weights_A) + 
                    criterion(outputs_B, y_B, weights_B)) / 2

        # Total loss
        loss = cls_loss + lambda_mmd * total_mmd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

        total_loss += loss.item()
        mmd_total += total_mmd.item()
        cls_total += cls_loss.item()

    avg_loss = total_loss / min_batches
    avg_mmd = mmd_total / min_batches
    avg_cls = cls_total / min_batches

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
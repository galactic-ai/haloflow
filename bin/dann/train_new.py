# %%
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import haloflow.data as D
from haloflow.config import get_dat_dir
from haloflow.dann.evalutate import evaluate
from haloflow.dann.model import DANNModel
from haloflow.dann.visualise import plot_evaluation_results

# %%
# set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)

# %%
# parse command line arguments
parser = argparse.ArgumentParser(description='Train a DANN model')
parser.add_argument('--obs', type=str, default='mags_morph_extra', help='Observation to use')
parser.add_argument('--sim', type=str, default='TNG50', help='Simulation to use')


args = parser.parse_args()
obs = args.obs
sim = args.sim

all_sims = ['TNG50', 'TNG100', 'Eagle100', 'Simba100']
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

MODEL_NAME = f'dann_model_to_{sim}_{obs}'
FP = get_dat_dir() + f'hf2/dann/models/{MODEL_NAME}.pt'

# %%
y, X, domains = [], [], []

# %%
for i, s in enumerate(all_sims):
    if s == sim:
        continue
    y_t, X_t = D.hf2_centrals("train", obs=obs, sim=s)

    domains.append(np.full(y_t.shape[0], i))
    y.append(y_t)
    X.append(X_t)

y = np.concatenate(y)
X = np.concatenate(X)
domains = np.concatenate(domains)

# standardize the data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# %%
# Create the model
input_dim = X.shape[1]
lr = 1e-2
num_epochs = 1000

model = DANNModel(input_dim).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
domain_criterion = nn.CrossEntropyLoss()

# Use AdamW optimizer with a learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=45, verbose=True)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
domains_tensor = torch.tensor(domains, dtype=torch.long).to(device)

# early stopping parameters
best_loss = float('inf')
patience = 50
counter = 0

# %%
losses = []

# %%
for epoch in range(num_epochs):
    model.train()

    #p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
    # alpha = 2. / (1. + np.exp(-10 * p)) - 1

    p = float(epoch) / num_epochs
    alpha = 2. / (1. + np.exp(-4.5 * p)) - 1
    
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs, domains = model(X_tensor, alpha)
    reg_loss = criterion(outputs, y_tensor) # regression loss
    domain_loss = domain_criterion(domains, domains_tensor) # domain classification loss

    # evalution
    _, _, eval_loss = evaluate(model, obs, sim, device=device)  # evaluation loss

    loss = reg_loss + domain_loss + eval_loss  # total loss


    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Step the learning rate scheduler
    scheduler.step(eval_loss)

    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}, Domain Loss: {domain_loss.item():.4f}, Eval Reg Loss: {eval_loss:.4f}, Alpha: {alpha:.4f}')

    # Early stopping
    if eval_loss < best_loss:
        best_loss = eval_loss
        counter = 0
        
        # Save the model checkpoint
        torch.save(model.state_dict(), FP)
    else:
        counter += 1
    
    if counter >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break

# %%
# Plot the training loss
plt.figure(figsize=(10, 5), dpi=150)
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid()
plt.savefig(f'../../plots/{MODEL_NAME}_training_loss.png')
plt.clf()

# %%
# Call the function to plot evaluation results
plot_evaluation_results(model, obs, sim, device, MODEL_NAME)

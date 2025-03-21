# %%
import numpy as np

from haloflow import data as D

# %%
obs = 'mags_morph_extra'
sim = 'TNG50'
all_sims = ['TNG50', 'TNG100', 'Eagle100', 'Simba100']

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

# %%
# standardize the data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# %%
import torch
import torch.nn as nn
import torch.optim as optim

# %%
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None



# %%
class DANNModel(nn.Module):
    def __init__(self, input_dim):
        super(DANNModel, self).__init__()
        self.feature = nn.Sequential()
        # self.feature.add_module('input', nn.Linear(input_dim, 512))
        # self.feature.add_module('silu_input', nn.SiLU())
        # self.feature.add_module('fc0', nn.Linear(input_dim, 256))
        # self.feature.add_module('silu0', nn.SiLU())
        self.feature.add_module('fc1', nn.Linear(input_dim, 128))
        self.feature.add_module('silu1', nn.SiLU())
        self.feature.add_module('fc2', nn.Linear(128, 64))
        self.feature.add_module('silu2', nn.SiLU())
        self.feature.add_module('fc3', nn.Linear(64, 32))
        self.feature.add_module('silu3', nn.SiLU())


        # class classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(32, 16))
        self.class_classifier.add_module('c_silu1', nn.SiLU())
        self.class_classifier.add_module('c_fc2', nn.Linear(16, 8))
        self.class_classifier.add_module('c_silu2', nn.SiLU())
        self.class_classifier.add_module('c_fc3', nn.Linear(8, 2)) # sm and hm

        # domain classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(32, 16))
        self.domain_classifier.add_module('d_silu1', nn.SiLU())
        self.domain_classifier.add_module('d_fc2', nn.Linear(16, 8))
        self.domain_classifier.add_module('d_silu2', nn.SiLU())
        self.domain_classifier.add_module('d_fc3', nn.Linear(8, 4)) # 4 domains
        

    def forward(self, x, alpha):
        x = self.feature(x)
        x_rev = ReverseLayerF.apply(x, alpha)
        label = self.class_classifier(x)
        domain = self.domain_classifier(x_rev)
        return label, domain

# %%
def evaluate(model, obs, sim, device='cpu'):
    """Evaluate the model on the test set."""
    # Load the test data
    y_eval, X_eval = D.hf2_centrals("test", obs=obs, sim=sim)
    X_eval = (X_eval - np.mean(X_eval, axis=0)) / np.std(X_eval, axis=0)
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred_tensor, _ = model(X_eval_tensor, 0)

    criterion = nn.MSELoss()
    y_eval_tensor = torch.tensor(y_eval, dtype=torch.float32).to(device)
    loss = criterion(y_pred_tensor, y_eval_tensor).item()
    
    y_pred = y_pred_tensor.cpu().numpy()
    y_eval = y_eval_tensor.cpu().numpy()

    return y_eval, y_pred, loss

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

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
# p = float(epoch) / num_epochs
# alpha = 2. / (1. + np.exp(-10 * p)) - 1
import matplotlib.pyplot as plt
import numpy as np

p = np.linspace(0, 1, num_epochs)
alpha = 2. / (1. + np.exp(-4.5 * p)) - 1

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(num_epochs), alpha, label='Alpha schedule', color='blue')

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
        torch.save(model.state_dict(), f'dann_model_to_{sim}_{obs}.pth')
    else:
        counter += 1
    
    if counter >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break

# %%
# Plot the training loss
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# %%
y_eval, y_pred, loss = evaluate(model, obs, 'TNG100', device)
print(f'Test Loss for {sim}: {loss:.4f}')

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(y_eval[:, 0], y_pred[:, 0], alpha=0.7, s=5.5)
ax[0].plot([10, 12.5], [10, 12.5], 'k--')
ax[0].set_xlabel('$M_*$')
ax[0].set_ylabel('Predicted $M_*$')
ax[0].set_xlim(10-0.3, 12.5+0.3)
ax[0].set_ylim(10-0.3, 12.5+0.3)

ax[1].scatter(y_eval[:, 1], y_pred[:, 1], alpha=0.7, s=5.5)
ax[1].plot([11.5, 15], [11.5, 15], 'k--')
ax[1].set_xlabel('$M_h$')
ax[1].set_ylabel('Predicted $M_h$')
ax[1].set_xlim(11.5-0.3, 15+0.3)
ax[1].set_ylim(11.5-0.3, 15+0.3)

print(f"MSE for stellar mass: {np.mean((y_eval[:, 0] - y_pred[:, 0])**2):.4f}")
print(f"MSE for halo mass: {np.mean((y_eval[:, 1] - y_pred[:, 1])**2):.4f}")

# %%




# %%
from haloflow.dann import data_loader as DL
from haloflow.config import get_dat_dir

# %%
from sklearn.metrics import mean_squared_error, r2_score

def test(model, dataloader):
    model.eval()
    
    # domain acc
    n_total = 0
    n_correct = 0
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for X, y, domain_label in dataloader:
            X = X.to('cpu')
            y = y.to('cpu')
            domain_label = domain_label.to('cpu')
            
            class_out, domain_out = model(X)
            
            y_true.append(y.cpu().numpy())
            y_pred.append(class_out.cpu().numpy())
            
            n_correct += (domain_out.argmax(dim=1) == domain_label).sum().item()
            n_total += len(domain_label)
    
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
            
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    domain_acc = n_correct / n_total
    
    return mse, rmse, r2, domain_acc

# %%
SIMS = ["Eagle100", "Simba100", "TNG100", "TNG50"]
dat_dir = get_dat_dir()

sim_data = DL.SimulationDataset(
    sims=SIMS,
    obs="mags",
    data_dir=dat_dir,
)
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
train_loader, test_loader = sim_data.get_train_test_loaders(
    train_sims=['Eagle100', 'TNG50', 'TNG100'],
    test_sim='Simba100',
    batch_size=32,
)

in_dim = next(iter(train_loader))[0].shape[1]

from collections import Counter

train_labels = []
test_labels = []

for _, _, train_label in train_loader:
    train_labels.append(train_label.cpu().numpy())

for _, _, test_label in test_loader:
    test_labels.append(test_label.cpu().numpy())

train_labels = np.concatenate(train_labels)
test_labels = np.concatenate(test_labels)

c_train = Counter(train_labels)
c_test = Counter(test_labels)

total_count = sum(c_train.values()) + sum(c_test.values())
weights = [c_train[i] / total_count for i in range(len(c_train))]
# append test
weights.append(c_test[3]/total_count)
weights = torch.tensor(weights, dtype=torch.float32).to(device)
weights

# So right now we have the training as Eagle100, TNG50, TNG100 
# and the testing as Simba100

# %%
from haloflow.dann import model as M

config = {
    "input_dim": in_dim,
    "num_domains": len(SIMS),
    "feature_layers": [128, 128, 64],
    "label_layers": [64, 32, 16],
    "domain_layers": [64, 16],
    "alpha": 0,
    "lr": 1e-2,
    "es_patience": 5,
    "es_min_delta": 1e-5,
    "num_epochs": 200,
}


model = M.build_from_config(M.DANN, config)
model

# %%
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from haloflow.dann import utils as U

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss functions
criterion_task = nn.MSELoss()  # For stellar/halo mass prediction (regression)
criterion_domain = nn.CrossEntropyLoss(weight=weights)  # For domain classification

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)

# Early stopping
early_stopper = U.EarlyStopper(
    patience=config["es_patience"],
    min_delta=config["es_min_delta"],
)

# Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=3, verbose=True
)

# %%
best_accu_t = 0.0 # accuracy on target set
for epoch in range(config["num_epochs"]):
    model.train()
    
    len_dataloader = min(len(train_loader), len(test_loader))
    data_source_iter = iter(train_loader)
    data_target_iter = iter(test_loader)
    
    for i in range(len_dataloader):
        p = float(i + epoch * len_dataloader) / config["num_epochs"] / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1.
        
        # model training on source
        data_source = next(data_source_iter)
        X, y, label = data_source
        
        model.zero_grad()
        batch_size = len(label)
        
        model.domain_classifier.grl.update_alpha(alpha)
        label_pred, domain_pred = model(X)
        
        err_s_task = criterion_task(label_pred, y)
        err_s_domain = criterion_domain(domain_pred, label)
        
        # model training on target
        data_target = next(data_target_iter)
        t_X, t_y, t_label = data_target
        
        batch_size = len(t_label)
        
        _, t_domain_pred = model(t_X)
        
        err_t_domain = criterion_domain(t_domain_pred, t_label)
        
        err = err_s_task + err_s_domain + err_t_domain
        
        err.backward()
        optimizer.step()
        scheduler.step(epoch)
        
        # print epoch, iteration, err_s_label, err_s_domain, err_t_domain, err
        sys.stdout.write(
            f"\rEpoch {epoch+1}/{config['num_epochs']}, Batch {i+1}/{len_dataloader}, Loss: {err.item():.4f}, Source Task: {err_s_task.item():.4f}, Source Domain: {err_s_domain.item():.4f}, Target Domain: {err_t_domain.item():.4f}"
        )
        sys.stdout.flush()
        # save ..
    
    print('\n')
    # mse, rmse, r2, acc_s = test(model, train_loader)
    # print(f"Domain Accuracy: {acc_s:}")
    mse, rmse, r2, acc_t = test(model, test_loader)
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    # break
    # 


 # %%
import matplotlib.pyplot as plt

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.scatter(y[:, 0], label_pred[:, 0].detach().numpy())
# plt.show()
model.eval()

sample_X, _, _ = next(iter(test_loader))

from haloflow.dann import get_preds
all_X, all_y = get_preds.get_all_data_from_loader(test_loader)

label_pred, _ = model(all_X)
scaler = sim_data.scaler_Y
all_y_pred = scaler.inverse_transform(label_pred.detach().numpy())
all_y_inv = scaler.inverse_transform(all_y.detach().numpy())

plt.figure(figsize=(10, 6))
plt.scatter(all_y_inv[:, 0], all_y_pred[:, 0])
plt.plot([10, 13], [10, 13], color='red', linestyle='--')
plt.xlim(10, 13)
plt.ylim(10, 13)
plt.show()

# %%
from haloflow.dann import data_loader as DL
import haloflow.config as C

# %%
sims = ['TNG50', 'TNG100', 'Eagle100', 'Simba100']
train_sims = sims[:-1]
test_sim = sims[-1]

# %%
dataset = DL.SimulationDataset(
    sims=sims,
    obs='mags',
    data_dir=C.get_dat_dir()
)

# %%
train_loader, test_loader = dataset.get_train_test_loaders(
    train_sims=train_sims,
    test_sim=test_sim,
    batch_size=64,
)


# %%
data = dataset.data

# %%
import numpy as np
train = np.concatenate([data['TNG50']["X_train"], data['TNG100']["X_train"], data['Eagle100']["X_train"]])
test = data['Simba100']["X_test"]

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)
# %%
train_inv = scaler.inverse_transform(train_scaled)
test_inv = scaler.inverse_transform(test_scaled)

assert np.allclose(train, train_inv)
assert np.allclose(test, test_inv)
test, test_inv

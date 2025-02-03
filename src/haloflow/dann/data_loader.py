import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from .. import data as D


class SimulationDataset:
    def __init__(self, sims, obs, data_dir):
        self.sims = sims
        self.obs = obs
        self.data_dir = data_dir
        self.data = self._load_data()

    def _load_data(self):
        data = {}
        for sim in self.sims:
            Y_train, X_train = D.hf2_centrals("train", self.obs, sim=sim)
            Y_test, X_test = D.hf2_centrals("test", self.obs, sim=sim)
            data[sim] = {
                "X_train": X_train,
                "Y_train": Y_train,
                "X_test": X_test,
                "Y_test": Y_test,
            }
        return data

    def get_train_test_loaders(self, train_sims, test_sim, batch_size=64):
        """Get DataLoaders for training and testing."""
        # Combine training data from specified simulations
        X_train = np.concatenate([self.data[sim]["X_train"] for sim in train_sims])
        Y_train = np.concatenate([self.data[sim]["Y_train"] for sim in train_sims])
        domain_labels = np.concatenate(
            [[i] * len(self.data[sim]["X_train"]) for i, sim in enumerate(train_sims)]
        )

        scaler = StandardScaler()

        # Get test data
        X_test = self.data[test_sim]["X_test"]
        Y_test = self.data[test_sim]["Y_test"]
        domain_labels_test = np.full(len(Y_test), len(train_sims))

        # Convert to tensors
        X_train_tensor = torch.tensor(
            scaler.fit_transform(X_train), dtype=torch.float32
        )
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        domain_labels_tensor = torch.tensor(domain_labels, dtype=torch.long)

        X_test_tensor = torch.tensor(scaler.fit_transform(X_test), dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
        domain_labels_tensor_test = torch.tensor(
            domain_labels_test, dtype=torch.long
        )

        # Create datasets
        train_dataset = TensorDataset(
            X_train_tensor, Y_train_tensor, domain_labels_tensor
        )
        test_dataset = TensorDataset(
            X_test_tensor, Y_test_tensor
        )

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

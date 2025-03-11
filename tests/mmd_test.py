import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

def mmd_loss(x, y, sigma=1.0):
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    X_sq = xx.diag().unsqueeze(1) + xx.diag().unsqueeze(0) - 2 * xx
    Y_sq = yy.diag().unsqueeze(1) + yy.diag().unsqueeze(0) - 2 * yy
    XY_sq = x.pow(2).sum(dim=1).unsqueeze(1) + y.pow(2).sum(dim=1).unsqueeze(0) - 2 * xy

    X_exp = torch.exp(-X_sq / (2 * sigma**2))
    Y_exp = torch.exp(-Y_sq / (2 * sigma**2))
    XY_exp = torch.exp(-XY_sq / (2 * sigma**2))

    loss = (X_exp.mean() + Y_exp.mean() - 2 * XY_exp.mean()) * 0.5
    return loss

# Hyperparameters
input_dim = 5  # feature dimension of the catalog data
hidden_dim = 128
output_dim = 2  # number of classes
lambda_mmd = 0.1  # weight for MMD loss
sigma = 1.0  # kernel bandwidth
lr = 0.0001
epochs = 100

# Initialize models
feature_extractor = FeatureExtractor(input_dim, hidden_dim)
classifier = Classifier(hidden_dim, output_dim)
optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=lr)

# Load data: separate loaders for each domain
from haloflow.dann.data_loader import SimulationDataset
sim_dataset = SimulationDataset(['TNG50', 'TNG100', 'Eagle100', 'Simba100'], 'mags', '../../data/hf2/')
loader_A, _ = sim_dataset.get_train_test_loaders(['TNG50'], 'Simba100')
loader_B, _ = sim_dataset.get_train_test_loaders(['TNG100'], 'Simba100')
loader_C, _ = sim_dataset.get_train_test_loaders(['Eagle100'], 'Simba100')

criterion = nn.MSELoss()

for epoch in range(epochs):
    total_loss = 0
    mmd_total = 0
    cls_total = 0

    for i, (batch_A, batch_B, batch_C) in enumerate(zip(loader_A, loader_B, loader_C)):
        x_A, y_A, _ = batch_A
        x_B, y_B, _ = batch_B
        x_C, y_C, _ = batch_C

        features_A = feature_extractor(x_A)
        features_B = feature_extractor(x_B)
        features_C = feature_extractor(x_C)

        # Compute MMD for all pairs
        mmd_AB = mmd_loss(features_A, features_B, sigma)
        mmd_AC = mmd_loss(features_A, features_C, sigma)
        mmd_BC = mmd_loss(features_B, features_C, sigma)
        total_mmd = mmd_AB + mmd_AC + mmd_BC

        # Compute classification loss
        outputs_A = classifier(features_A)
        outputs_B = classifier(features_B)
        outputs_C = classifier(features_C)
        cls_loss = (criterion(outputs_A, y_A) + 
                    criterion(outputs_B, y_B) + 
                    criterion(outputs_C, y_C)) / 3

        # Total loss
        loss = cls_loss + lambda_mmd * total_mmd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        mmd_total += total_mmd.item()
        cls_total += cls_loss.item()

    avg_loss = total_loss / (i + 1)
    avg_mmd = mmd_total / (i + 1)
    avg_cls = cls_total / (i + 1)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, MMD Loss={avg_mmd:.4f}, CLS Loss={avg_cls:.4f}")


# Evaluate on test domain
_, test_loader = sim_dataset.get_train_test_loaders(['Simba100'], 'Simba100')
feature_extractor.eval()

total_loss = 0
total_samples = 0

predicted_values = []
true_values = []

features_list = []


criterion = nn.MSELoss()  # Since it's a regression task

with torch.no_grad():
    for x_test, y_test, _ in test_loader:
        # Extract features and classify
        features = feature_extractor(x_test)
        features_list.append(features.numpy())
        
        outputs = classifier(features)
        
        predicted_values.append(outputs.numpy())
        true_values.append(y_test.numpy())

        # Calculate loss (Mean Squared Error)
        loss = criterion(outputs, y_test)
        
        total_loss += loss.item() * x_test.size(0)  # Accumulate the loss weighted by batch size
        total_samples += x_test.size(0)  # Accumulate the total number of samples

# Calculate average loss over all samples
avg_loss = total_loss / total_samples
print(f"Test MSE Loss: {avg_loss:.4f}")

# Convert to numpy arrays
predicted_values = np.concatenate(predicted_values, axis=0)
true_values = np.concatenate(true_values, axis=0)
features_array = np.concatenate(features_list, axis=0)

# Plot the predicted vs true values for stellar and halo masses
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(true_values[:, 0], predicted_values[:, 0], color='blue', alpha=0.5)
plt.plot([10, 13], [10, 13], 'k--')
plt.xlim(10, 13)
plt.ylim(10, 13)
plt.xlabel('True Stellar Mass')
plt.ylabel('Predicted Stellar Mass')
plt.title('Stellar Mass Predictions')

plt.subplot(1, 2, 2)
plt.scatter(true_values[:, 1], predicted_values[:, 1], color='red', alpha=0.5)
plt.plot([11, 15], [11, 15], 'k--')
plt.xlim(11.5, 15)
plt.ylim(11.5, 15)
plt.xlabel('True Halo Mass')
plt.ylabel('Predicted Halo Mass')
plt.title('Halo Mass Predictions')

plt.tight_layout()
plt.show()

# umap
import umap
reducer = umap.UMAP()
embedding = reducer.fit_transform(features_array)
plt.figure(figsize=(10, 5))
plt.scatter(embedding[:, 0], embedding[:, 1], c=true_values[:, 0], cmap='viridis', alpha=0.5)
plt.colorbar(label='True Stellar Mass')
plt.title('UMAP Embedding of Features')
plt.show()

import torch
import umap
import matplotlib.pyplot as plt
import numpy as np

# Assuming the feature extractor and data loaders for all domains are defined earlier

# Set the model to evaluation mode for feature extraction
feature_extractor.eval()

# Initialize lists to hold the features and the domain labels
features_list = []
domain_labels = []

# Loop through each domain's loader to get the features
with torch.no_grad():
    for domain_name, loader in zip([0, 1, 2], [loader_A, loader_B, loader_C]):
        for x_data, y_data, _ in loader:
            # Extract features for the current domain
            features = feature_extractor(x_data)

            # Collect features for UMAP
            features_list.append(features.numpy())

            # Collect domain labels (0 for TNG50, 1 for TNG100, 2 for Eagle100)
            domain_labels.append(np.full(features.shape[0], domain_name))  # Label by domain

# Convert features and labels to numpy arrays
features_array = np.concatenate(features_list, axis=0)
domain_labels = np.concatenate(domain_labels, axis=0)

# Apply UMAP for dimensionality reduction (2D)
reducer = umap.UMAP()
embedding = reducer.fit_transform(features_array)

# Plot UMAP embedding with domain labels as color
plt.figure(figsize=(10, 5))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=domain_labels, cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Domain')
plt.title('UMAP Embedding of Features (Colored by Domain)')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.show()



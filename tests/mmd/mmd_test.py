import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from haloflow.mmd.models import MMDModel
from haloflow.mmd.utils import mmd_loss

# See bin/mmd/train and bin/mmd/eval

# # umap
# import umap
# reducer = umap.UMAP()
# embedding = reducer.fit_transform(features_array)
# plt.figure(figsize=(10, 5))
# plt.scatter(embedding[:, 0], embedding[:, 1], c=true_values[:, 0], cmap='viridis', alpha=0.5)
# plt.colorbar(label='True Stellar Mass')
# plt.title('UMAP Embedding of Features')
# plt.show()

# import torch
# import umap
# import matplotlib.pyplot as plt
# import numpy as np

# # Assuming the feature extractor and data loaders for all domains are defined earlier

# # Set the model to evaluation mode for feature extraction
# feature_extractor.eval()

# # Initialize lists to hold the features and the domain labels
# features_list = []
# domain_labels = []

# # Loop through each domain's loader to get the features
# with torch.no_grad():
#     for domain_name, loader in zip([0, 1, 2], [loader_A, loader_B, loader_C]):
#         for x_data, y_data, _ in loader:
#             # Extract features for the current domain
#             features = feature_extractor(x_data)

#             # Collect features for UMAP
#             features_list.append(features.numpy())

#             # Collect domain labels (0 for TNG50, 1 for TNG100, 2 for Eagle100)
#             domain_labels.append(np.full(features.shape[0], domain_name))  # Label by domain

# # Convert features and labels to numpy arrays
# features_array = np.concatenate(features_list, axis=0)
# domain_labels = np.concatenate(domain_labels, axis=0)

# # Apply UMAP for dimensionality reduction (2D)
# reducer = umap.UMAP()
# embedding = reducer.fit_transform(features_array)

# # Plot UMAP embedding with domain labels as color
# plt.figure(figsize=(10, 5))
# scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=domain_labels, cmap='viridis', alpha=0.5)
# plt.colorbar(scatter, label='Domain')
# plt.title('UMAP Embedding of Features (Colored by Domain)')
# plt.xlabel('UMAP Component 1')
# plt.ylabel('UMAP Component 2')
# plt.tight_layout()
# plt.show()



from haloflow.mmd.models import MMDModel
from haloflow.dann.data_loader import SimulationDataset
from haloflow.config import get_dat_dir 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

SIMS = ['TNG50', 'TNG100', 'Eagle100', 'Simba100']
OBS = 'mags'
TEST_SIM = 'Simba100'
TRAIN_SIMS = [sim for sim in SIMS if sim != TEST_SIM]

FP = get_dat_dir() + f'hf2/mmd/models/mmd_best_model_to_{TEST_SIM}_{OBS}.pth'
# FP = get_dat_dir() + f'hf2/mmd/models/mmd_best_model_to_TNG50_{OBS}.pth'

sim_dataset = SimulationDataset(SIMS, OBS, get_dat_dir())
_, test_loader = sim_dataset.get_train_test_loaders(TRAIN_SIMS, TEST_SIM)
# scaler = sim_dataset.scaler_Y

input_dim = next(iter(test_loader))[0].shape[1]
output_dim = 2

mmd_model = MMDModel(input_dim, output_dim)
mmd_model.to('cpu')
mmd_model.load_state_dict(torch.load(FP))
mmd_model.eval()

criterion = nn.MSELoss()

total_loss = 0
total_samples = 0

predicted_values = []
true_values = []

features_list = []

with torch.no_grad():
    for x_test, y_test, _ in test_loader:
        features, outputs = mmd_model(x_test)
        features_list.append(features.numpy())
        
        predicted_values.append(outputs.numpy())
        true_values.append(y_test.numpy())

        # Calculate loss (Mean Squared Error)
        loss = criterion(outputs, y_test)
        
        # print loss for separate outputs
        print(criterion(outputs[:, 0], y_test[:, 0]), criterion(outputs[:, 1], y_test[:, 1]))
        
        # inverse transform the predicted values
        # predicted_values[-1] = scaler.inverse_transform(predicted_values[-1])
        # true_values[-1] = scaler.inverse_transform(true_values[-1])
        
        total_loss += loss.item()
        total_samples += x_test.size(0)  # Accumulate the total number of samples

# Calculate average loss over all samples
avg_loss = total_loss / total_samples
print(f"Test MSE Loss: {avg_loss:.4f}")

# Convert to numpy arrays
predicted_values = np.concatenate(predicted_values, axis=0)
true_values = np.concatenate(true_values, axis=0)
features_array = np.concatenate(features_list, axis=0)

# Plot the predicted vs true values for stellar and halo masses
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(true_values[:, 0], predicted_values[:, 0], color='blue', alpha=0.5, s=3)
# ax[0].plot([np.min(true_values[:, 0]), np.max(true_values[:, 0])], [np.min(true_values[:, 0]), np.max(true_values[:, 0])], 'k--')
ax[0].plot([10, 13], [10, 13], 'k--')
ax[0].set_xlim(10-0.05, 13+0.05)
ax[0].set_ylim(10-0.05, 13+0.05)
ax[0].set_xlabel('True Stellar Mass')
ax[0].set_ylabel('Predicted Stellar Mass')
ax[0].set_title('Stellar Mass Predictions')

ax[1].scatter(true_values[:, 1], predicted_values[:, 1], color='red', alpha=0.5, s=3)
# ax[1].plot([np.min(true_values[:, 1]), np.max(true_values[:, 1])], [np.min(true_values[:, 1]), np.max(true_values[:, 1])], 'k--')
ax[1].plot([11, 15], [11, 15], 'k--')
ax[1].set_xlim(11.5-0.05, 15+0.05)
ax[1].set_ylim(11.5-0.05, 15+0.05)
ax[1].set_xlabel('True Halo Mass')
ax[1].set_ylabel('Predicted Halo Mass')
ax[1].set_title('Halo Mass Predictions')

plt.tight_layout()
plt.show()

# import shap

# # Convert PyTorch model into a SHAP-compatible function
# def model_wrapper(input_data):
#     tensor_input = torch.tensor(input_data, dtype=torch.float32)  # Convert input to tensor
#     with torch.no_grad():
#         features, outputs = mmd_model(tensor_input)  # Extract outputs from model
#     return outputs.numpy()  # Convert to NumPy for SHAP

# # Select a small sample for SHAP analysis
# sample_data = next(iter(test_loader))[0][:100].numpy()  # Convert sample to NumPy

# # Create the SHAP explainer using the modified model function
# explainer = shap.Explainer(model_wrapper, sample_data)

# # Compute SHAP values
# shap_values = explainer(sample_data)

# # Plot feature importance
# shap.summary_plot(shap_values, sample_data)

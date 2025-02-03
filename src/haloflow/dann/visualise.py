import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

def visualize_features_tsne(model, dataloader, device="cuda", target_domain=4):
    """
    Visualize the feature space using t-SNE.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    dataloader : torch.utils.data.DataLoader
        DataLoader for the data.
    device : str
        Device to run visualization on.
    target_domain : int
        Domain to visualize (default: 4).
    
    Returns
    -------
    None
    """
    model.eval()
    features, domains = [], []

    with torch.no_grad():
        for batch in dataloader:
            # Check if domain labels exist
            if len(batch) == 3:
                X_batch, _, domain_batch = batch
            else:
                X_batch, _ = batch
                domain_batch = torch.full((X_batch.shape[0],), target_domain, dtype=torch.long)
            
            X_batch = X_batch.to(device)
            feats = model.feature_extractor(X_batch).cpu().numpy()
            features.append(feats)
            domains.append(domain_batch.cpu().numpy())

    features = np.concatenate(features)
    domains = np.concatenate(domains)

    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(features)

    # Plot; TODO: Add more customization and axes return
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(embeddings[:,0], embeddings[:,1], c=domains, 
                          cmap='tab10', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Domains")
    plt.title("Feature Space Visualization (t-SNE)")
    plt.show()

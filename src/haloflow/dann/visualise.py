import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

from .. import config as C

C.setup_plotting_config()

def visualize_features_fast(model, train_loader, test_loader, n_samples=2000, test_domain_label=4, device="cuda"):
    """
    Visualize features using t-SNE. This function is optimized for speed.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training set.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test set.
    n_samples : int
        Number of samples to visualize.
    test_domain_label : int
        Domain label for the test set.
    device : str
        Device to run visualization on.
    
    Returns
    -------
    np.ndarray
        t-SNE embeddings.
    np.ndarray
        Domain labels.
    """
    model.eval()
    features, domains = [], []

    # Process training data
    with torch.no_grad():
        for X_batch, _, domain_batch in train_loader:
            X_batch = X_batch.to(device)
            feats = model.feature_extractor(X_batch).cpu().numpy()
            features.append(feats)
            domains.append(domain_batch.cpu().numpy())

    # Process test data (assign domain=4)
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            feats = model.feature_extractor(X_batch).cpu().numpy()
            features.append(feats)
            domains.append(np.full(feats.shape[0], test_domain_label))  # Test domain label=4

    features = np.concatenate(features)
    domains = np.concatenate(domains)

    # Subsample
    idx = np.random.choice(len(features), n_samples, replace=False)
    features = features[idx]
    domains = domains[idx]

    # t-SNE with faster parameters
    tsne = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=42)
    embeddings = tsne.fit_transform(features)
    
    return embeddings, domains


def plot_combined_tsne(embeddings, domains, train_domains=[0,1,2,3], test_domain=4):
    """
    Plot t-SNE embeddings of the feature space.

    Parameters
    ----------
    embeddings : np.ndarray
        t-SNE embeddings.
    domains : np.ndarray
        Domain labels.
    train_domains : list
        List of training domain labels.
    test_domain : int
        Test domain label.
    
    Returns
    -------
    None
    """
    plt.figure(figsize=(12, 8))
    
    # Plot training domains (0-3)
    for domain in train_domains:
        mask = (domains == domain)
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                    label=f"Train Domain {domain}", alpha=0.6)

    # Plot test domain (4)
    mask = (domains == test_domain)
    plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                color='black', marker='x', label="Test Domain", alpha=0.6)

    plt.legend()
    # plt.title("t-SNE of Feature Space (Train vs Test)")
    plt.show()
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from .. import config as C
from .evalutate import evaluate

C.setup_plotting_config()


def visualize_features_fast(
    model, train_loader, test_loader, n_samples=2000, device="cuda"
):
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
            X_batch = X_batch.to(device).float()
            feats = model.feature(X_batch).cpu().numpy()
            features.append(feats)
            domains.append(domain_batch.cpu().numpy())

    # Process test data (assign domain=4)
    with torch.no_grad():
        for X_batch, _, domain_batch in test_loader:
            X_batch = X_batch.to(device).float()
            feats = model.feature(X_batch).cpu().numpy()
            features.append(feats)
            domains.append(domain_batch.cpu().numpy())

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


def plot_combined_tsne(embeddings, domains, train_domains=[0, 1, 2, 3], test_domain=4):
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
    fig = plt.figure(figsize=(12, 8))

    # Plot training domains (0-3)
    for domain in train_domains:
        mask = domains == domain
        plt.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            label=f"Train Domain {domain}",
            alpha=0.6,
        )

    # Plot test domain (4)
    mask = domains == test_domain
    plt.scatter(
        embeddings[mask, 0],
        embeddings[mask, 1],
        color="black",
        marker="x",
        label="Test Domain",
        alpha=0.6,
    )

    plt.legend()
    # plt.title("t-SNE of Feature Space (Train vs Test)")
    #     plt.show()
    return fig


def plot_evaluation_results(model, obs, sim, device, model_name, mean_, std_, weights):
    y_eval, y_pred, loss, _ = evaluate(model, obs, sim, device=device, mean_=mean_, std_=std_, weights=None)
    print(f"Test Loss for {sim}: {loss:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

    plt.suptitle(f"Predictions for Sims - {sim} with {obs} obs")

    ax[0].scatter(y_eval[:, 0], y_pred[:, 0], alpha=0.7, s=5.5)
    ax[0].plot([10, 12.5], [10, 12.5], "k--")
    # make a 1:1 line 0.3 dex above and below the 1:1 line
    ax[0].fill_between(
        [10, 12.5], [10 - 0.3, 12.5 - 0.3], [10 + 0.3, 12.5 + 0.3], color="gray", alpha=0.2
    )
    
    ax[0].set_xlabel("$M_*$", fontsize='x-large')
    ax[0].set_ylabel("Predicted $M_*$", fontsize='x-large')
    ax[0].set_xlim(10 - 0.3, 12.5 + 0.3)
    ax[0].set_ylim(10 - 0.3, 12.5 + 0.3)

    ax[1].scatter(y_eval[:, 1], y_pred[:, 1], alpha=0.7, s=5.5)
    ax[1].plot([11.5, 15], [11.5, 15], "k--")
    # make a 1:1 line 0.3 dex above and below the 1:1 line
    ax[1].fill_between(
        [11.5, 15], [11.5 - 0.3, 15 - 0.3], [11.5 + 0.3, 15 + 0.3], color="gray", alpha=0.2
    )
    ax[1].set_xlabel("$M_h$", fontsize='x-large')
    ax[1].set_ylabel("Predicted $M_h$", fontsize='x-large')
    ax[1].set_xlim(11.5 - 0.3, 15 + 0.3)
    ax[1].set_ylim(11.5 - 0.3, 15 + 0.3)

    plt.tight_layout()
    plt.savefig(f"../../plots/{model_name}_predictions.png")

    with open(f"../../plots/{model_name}_mse_results.txt", "w") as f:
        f.write(
            f"MSE for stellar mass: {np.mean((y_eval[:, 0] - y_pred[:, 0]) ** 2):.4f}\n"
        )
        f.write(
            f"MSE for halo mass: {np.mean((y_eval[:, 1] - y_pred[:, 1]) ** 2):.4f}\n"
        )

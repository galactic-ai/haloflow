from haloflow import data as D
from haloflow import util as U
import numpy as np
import torch
from tarp import get_tarp_coverage

def validate_npe(train_obs, train_sim, 
                test_obs, test_sim, 
                device='cpu',
                data_dir='/xdisk/chhahn/chhahn/haloflow/hf2/npe', 
                n_ensemble=5,
                n_samples=10_000,
                train_samples=None,
                version=1,
                with_dann=False):
    """
    Function to validate the NDEs trained on the training set
    on the test set. This function returns the ranks of the
    test set in the training set, the alpha and ECP of the
    NDEs on the test set.
    """

    # Load NDEs
    if with_dann:
        qphis = U.read_best_ndes(
            f'h2.dann.v{version}.{train_sim}.{train_obs}',
            n_ensemble=n_ensemble, device=device,
            dat_dir=data_dir, verbose=True)
    else:
        qphis = U.read_best_ndes(
            f'h2.v{version}.{train_sim}.{train_obs}',
            n_ensemble=n_ensemble, device=device,
            dat_dir=data_dir, verbose=True)

    # Load test data
    Y_test, X_test = D.hf2_centrals('test', test_obs, sim=test_sim, version=version)
    
    # Select subset if needed
    if train_samples is not None:
        np.random.seed(42)
        idx = np.random.choice(len(Y_test), train_samples, replace=False)
        Y_test = Y_test[idx]
        X_test = X_test[idx]

    # Convert test data to tensors once (on GPU if available)
    Y_test_torch = torch.tensor(Y_test, dtype=torch.float32, device=device)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32, device=device)

    # Pre-allocate memory
    num_test_samples = len(Y_test)
    ranks = np.empty((num_test_samples, Y_test.shape[1]), dtype=np.float32)
    y_nde = np.empty((num_test_samples, n_samples, Y_test.shape[1]), dtype=np.float32)

    # Sample in batches to optimize performance
    for i in range(num_test_samples):
        y_samp = torch.cat([
            qphi.sample((n_samples // len(qphis),), x=X_test_torch[i], show_progress_bars=False)
            for qphi in qphis
        ], dim=0)  # Collect all samples at once

        # Compute ranks in a vectorized way
        ranks[i] = (y_samp < Y_test_torch[i]).float().mean(dim=0).cpu().numpy()

        # Store samples efficiently
        y_nde[i] = y_samp.cpu().numpy()

    # Calculate TARP coverages
    ecp, alpha = get_tarp_coverage(
        np.swapaxes(y_nde, 0, 1),
        Y_test,
        references="random",
        metric="euclidean"
    )

    return ranks, alpha, ecp, y_nde

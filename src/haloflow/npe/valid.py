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
                n_samples = 10_000,
                train_samples=None,
                version=1):
    """
    Function to validate the NDEs trained on the training set
    on the test set. This function returns the ranks of the
    test set in the training set, the alpha and ECP of the
    NDEs on the test set.
    """

    # Load the NDEs trained on the training set
    qphis = U.read_best_ndes(
        f'h2.v1.{train_sim}.{train_obs}',
        n_ensemble=n_ensemble, device=device,
        dat_dir=data_dir,
        verbose=True)
    
    # read test data
    Y_test, X_test = D.hf2_centrals('test', test_obs, sim=test_sim, version=version)


    # choose a subset of the test set
    if train_samples is not None:
        np.random.seed(42)
        idx = np.random.choice(np.arange(Y_test.shape[0]), train_samples, replace=False)
        # Y_test = Y_test[idx]
        # X_test = X_test[idx]

        # make sure the samples are used in the same order
        train_samples_len = idx
    else:
        train_samples_len = range(len(Y_test))

    ranks = []
    y_nde = []

    # move X_test to device once
    Y_test_torch = torch.tensor(Y_test, dtype=torch.float32).to(device)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)

    for i in train_samples_len:
        # sample in GPU and concatenate
        y_samp = []
        for qphi in qphis:
            _samp = qphi.sample(
                (int(n_samples / len(qphis)),),
                x=X_test_torch[i],
                show_progress_bars=False
            )
            y_samp.append(_samp)
        y_cat = torch.cat(y_samp, dim=0)

        # compute ranks in a vectorized way
        ranks.append((y_cat < Y_test_torch[i]).float().mean(dim=0).cpu().numpy())

        # store samples in CPU
        y_nde.append(y_cat.cpu().numpy())

    ranks = np.array(ranks)
    y_nde = np.array(y_nde)

    if train_samples is not None:
        Y_test = Y_test[idx]
        
    # calculate TARP coverages
    ecp, alpha = get_tarp_coverage(
        np.swapaxes(y_nde, 0, 1),
        Y_test,
        references="random",
        metric="euclidean",
    )
    # if we draw from same prior, posteriors dervied from the same prior
    # should have the same ECP and alpha

    return ranks, alpha, ecp, y_nde
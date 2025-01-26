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
    
    Y_test, X_test = D.hf2_centrals('test', test_obs, sim=test_sim, version=version)

    ranks = []
    y_nde = []
    for i in range(X_test.shape[0]):
        y_samp = []
        x_tensor = torch.tensor(X_test[i], dtype=torch.float32).to(device)
        for qphi in qphis:
            _samp = qphi.sample(
                (int(n_samples/len(qphis)),),
                x=x_tensor,
                show_progress_bars=False
            )
            y_samp.append(_samp.detach().cpu().numpy())
        y_nde.append(np.concatenate(y_samp, axis=0))

        _ranks = []
        for i_dim in range(y_nde[-1].shape[1]):
            _ranks.append(np.mean(y_nde[-1][:,i_dim].flatten() < Y_test[i, i_dim]))
        ranks.append(_ranks)

    ranks = np.array(ranks)
    y_nde = np.array(y_nde)

    ecp, alpha = get_tarp_coverage(
        np.swapaxes(y_nde, 0, 1),
        Y_test,
        references="random",
        metric="euclidean",
    )

    return ranks, alpha, ecp
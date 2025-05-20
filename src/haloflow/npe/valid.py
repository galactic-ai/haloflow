from haloflow import data as D
from haloflow import util as U

from haloflow.config import get_dat_dir
from haloflow.dann.evalutate import evaluate
from haloflow.dann.model import DANNModel

import numpy as np
import torch
from tarp import get_tarp_coverage

def validate_npe(
        npe_train_obs,
        dann_sim,
        npe_train_sim,
        test_obs,
        test_sim,
        device='cpu',
        data_dir='/xdisk/chhahn/chhahn/haloflow/hf2/npe',
        n_ensemble=5,
        n_samples=1_000,
        train_samples=None,
        version=1,
        with_dann=False,
):

    # Load test data
    y_test, x_test = D.hf2_centrals('test', test_obs, sim=test_sim, version=version)

    if with_dann:
        MODEL_NAME = f'dann_model_v3_to_{dann_sim}_{test_obs}'
        FP = get_dat_dir() + f'hf2/dann/models/{MODEL_NAME}.pt'
        FP_mean_std = get_dat_dir() + f'hf2/dann/models/{MODEL_NAME}_mean_std.npz'
        study_name = f'h2.dann.v3.m{dann_sim}.{npe_train_sim}.{npe_train_obs}'


        input_dim = x_test.shape[1]
        model = DANNModel(input_dim=input_dim).to(device)
        model.load_state_dict(torch.load(FP, map_location=device))

        array = np.load(FP_mean_std)
        mean_, std_ = array['mean'], array['std']

        y_test, x_test, _, _ = evaluate(
            model,
            test_obs,
            test_sim,
            device=device,
            dataset='test',
            mean_=mean_,
            std_=std_,
        )

    if with_dann:
        qphis = U.read_best_ndes(
            study_name,
            n_ensemble=n_ensemble,
            device=device,
            dat_dir=data_dir,
            verbose=True,
        )
    else:
        qphis = U.read_best_ndes(
            f'h2.v{version}.{npe_train_sim}.{npe_train_obs}',
            n_ensemble=n_ensemble, device=device,
            dat_dir=data_dir, verbose=True)

    if train_samples is not None:
        np.random.seed(42)
        idx = np.random.choice(len(y_test), train_samples, replace=False)
        y_test = y_test[idx]
        x_test = x_test[idx]

    Y_test_torch = torch.tensor(y_test, dtype=torch.float32, device=device)
    X_test_torch = torch.tensor(x_test, dtype=torch.float32, device=device)

    # Pre-allocate memory
    num_test_samples = len(Y_test_torch)
    ranks = np.empty((num_test_samples, Y_test_torch.shape[1]), dtype=np.float32)
    y_nde = np.empty((num_test_samples, n_samples, Y_test_torch.shape[1]), dtype=np.float32)

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
        y_test, # needs to be np.ndarray
        references="random",
        metric="euclidean"
    )

    return ranks, alpha, ecp, y_nde

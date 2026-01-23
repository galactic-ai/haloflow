from .. import data as D
from .. import util as U

from ..config import get_dat_dir
from ..dann.evalutate import evaluate
from ..dann.model import DANNModel

import numpy as np
import torch
from tarp import get_tarp_coverage
from tqdm.auto import tqdm

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
        with_mmd=False,
):

    # Load test data
    y_test, x_test = D.hf2_centrals('test', test_obs, sim=test_sim, version=version)

    if with_dann:
        MODEL_NAME = f'dann_model_v3_to_{dann_sim}_{test_obs}'
        FP = get_dat_dir() + f'hf2/dann/models/{MODEL_NAME}.pt'
        FP_mean_std = get_dat_dir() + f'hf2/dann/models/{MODEL_NAME}_mean_std.npz'
        study_name = f'h2.dann.v3.m{dann_sim}.{npe_train_sim}.{npe_train_obs}.feature.extract'


        input_dim = x_test.shape[1]
        model = DANNModel(input_dim=input_dim).to(device)
        model.load_state_dict(torch.load(FP, map_location=device))

        array = np.load(FP_mean_std)
        mean_, std_ = array['mean'], array['std']

        y_test, _, _, _, x_test = evaluate(
            model,
            test_obs,
            test_sim,
            device=device,
            dataset='test',
            mean_=mean_,
            std_=std_,
        )
    elif with_mmd:
        from haloflow.mmd.get_preds import get_mmd_preds

        MODEL_NAME = f'mmd_model_v2_to_{dann_sim}_{test_obs}'
        FP = get_dat_dir() + f'hf2/mmd/models/{MODEL_NAME}.pt'
        study_name = f'h2.mmd.v2.m{dann_sim}.{npe_train_sim}.{npe_train_obs}'

        _, x_test = get_mmd_preds(FP, test_obs, test_sim)

    if with_dann or with_mmd:
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
        # instance is a number of samples then do random sampling
        if isinstance(train_samples, int):
            idx = np.random.choice(len(y_test), train_samples, replace=False)
        # instance is a list or numpy array of indices
        elif isinstance(train_samples, (list, np.ndarray)):
            idx = train_samples
        else:
            raise ValueError("train_samples must be an int, list, or np.ndarray")
        y_test = y_test[idx]
        x_test = x_test[idx]

    Y_test_torch = torch.tensor(y_test, dtype=torch.float32, device=device)
    X_test_torch = torch.tensor(x_test, dtype=torch.float32, device=device)

    # Pre-allocate memory
    num_test_samples = len(Y_test_torch)
    ranks = np.empty((num_test_samples, Y_test_torch.shape[1]), dtype=np.float32)
    y_nde = np.empty((num_test_samples, n_samples, Y_test_torch.shape[1]), dtype=np.float32)

    # Sample in batches to optimize performance
    with tqdm(total=num_test_samples, desc="Validating NPE") as pbar:
        for i in range(num_test_samples):
            y_samp = torch.cat([
                qphi.sample((n_samples // len(qphis),), x=X_test_torch[i], show_progress_bars=False)
                for qphi in qphis
            ], dim=0)  # Collect all samples at once

            # Compute ranks in a vectorized way
            ranks[i] = (y_samp < Y_test_torch[i]).float().mean(dim=0).cpu().numpy()

            # Store samples efficiently
            y_nde[i] = y_samp.cpu().numpy()
            pbar.update(1)

    # Calculate TARP coverages
    ecp, alpha = get_tarp_coverage(
        np.swapaxes(y_nde, 0, 1),
        y_test, # needs to be np.ndarray
        references="random",
        metric="euclidean"
    )

    return ranks, alpha, ecp, y_nde

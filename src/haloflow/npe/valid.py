from .. import data as D
from .. import util as U
from .. import corr as Corr

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


def standard_method(train_sim, test_sim):
    """
    Standard method to get HM predictions from SM predictions 
    using the SMHR from the training simulation.

    Method:
    1. Get the SM-HM relation from the training simulation.
    2. Get the NDE samples (using the training simulation) for the test simulation.
    3. For each NDE sample, get the corresponding HM prediction using the SM-HM relation.
    4. Return the HM predictions for the test simulation.

        Note: This method does not use any domain adaptation techniques and serves as a baseline for comparison.

    Parameters
    ----------
    train_sim : str
        The name of the training simulation (e.g., 'Simba100').
    test_sim : str
        The name of the test simulation (e.g., 'Eagle100').
    
    Returns
    -------
    y_test : np.ndarray
        The true values of the test simulation.
    y_nde_test_mags : np.ndarray
        The NDE samples for the test simulation. (nde using train_sim with mags only)
    hm_train_SMHR_test : np.ndarray
        The HM predictions for the test simulation using the 
        SM-HM relation from the training simulation added to the NDE samples.
    """

    rng = np.random.default_rng(42)

    hm_bins = np.linspace(11.5, 14.1, 10)
    hm_bin_centers = 0.5 * (hm_bins[1:] + hm_bins[:-1])

    sm_bins = np.arange(10, 12.1, 0.25)
    sm_bin_centers = 0.5 * (sm_bins[1:] + sm_bins[:-1])

    # build grid for train SMHR
    y_train, _ = D.hf2_centrals('test', 'mags', sim=train_sim, version=1)
    sm_train = y_train[:, 0]
    hm_train = y_train[:, 1]

    median_hm_train = []
    sig_hm_train = []

    for lo, hi in zip(sm_bins[:-1], sm_bins[1:]):
        in_bin = (sm_train >= lo) & (sm_train < hi)
        if in_bin.any():
            median_hm_train.append(np.median(hm_train[in_bin]))
            q16, q84 = np.quantile(hm_train[in_bin], [0.16, 0.84])
            sig_hm_train.append(0.5 * (q84 - q16))
        else:
            median_hm_train.append(np.nan)
            sig_hm_train.append(np.nan)
    
    median_hm_train = np.array(median_hm_train)
    sig_hm_train = np.array(sig_hm_train)

    m = np.isfinite(median_hm_train) & np.isfinite(sig_hm_train)
    s_grid = sm_bin_centers[m]
    mu_grid = median_hm_train[m]
    sig_grid = sig_hm_train[m]

    # p_train(theta|X_test, mags); theta is just sm here
    qphis = U.read_best_ndes(
        f'h2.v1.{train_sim}.mags',
        n_ensemble=5, device='cpu',
        dat_dir=get_dat_dir() + 'hf2/npe', verbose=True
    )
    n_samples = 1000

    # X_test is mags
    y_test, X_test = D.hf2_centrals('test', 'mags', sim=test_sim, version=1)

    y_test_torch = torch.tensor(y_test, dtype=torch.float32, device='cpu')
    X_test_torch = torch.tensor(X_test, dtype=torch.float32, device='cpu')

    # Pre-allocate memory
    num_test_samples = len(y_test_torch)
    ranks = np.empty((num_test_samples, y_test_torch.shape[1]), dtype=np.float32)
    y_nde_test_mags_unweighted = np.empty((num_test_samples, n_samples, y_test_torch.shape[1]), dtype=np.float32)

    # Sample in batches to optimize performance
    with tqdm(total=num_test_samples, desc=f"Validating NPE for {test_sim}") as pbar:
        for i in range(num_test_samples):
            y_samp = torch.cat([
                qphi.sample((n_samples // len(qphis),), x=X_test_torch[i], show_progress_bars=False)
                for qphi in qphis
            ], dim=0)  # Collect all samples at once

            # Compute ranks in a vectorized way
            ranks[i] = (y_samp < y_test_torch[i]).float().mean(dim=0).cpu().numpy()

            # Store samples efficiently
            y_nde_test_mags_unweighted[i] = y_samp.cpu().numpy()
            pbar.update(1)

    # weight the nde samples
    y_nde_test_mags = Corr.weight_nde(y_nde_test_mags_unweighted, train_sim)
    sm_nde_test_mags = y_nde_test_mags[:, :, 0]

    # get the posterior median and 1-sigma intervals
    # y_nde_test_mags_q0, y_nde_test_mags_q1, y_nde_test_mags_q2 = np.quantile(y_nde_test_mags, (0.16, 0.5, 0.84), axis=1)

    mu_h = np.interp(sm_nde_test_mags, s_grid, mu_grid)
    std_h = np.interp(sm_nde_test_mags, s_grid, sig_grid)
    hm_train_SMHR_test = mu_h + rng.normal(0.0, std_h, size=sm_nde_test_mags.shape)

    # get the posterior median and 1-sigma intervals for the HM predictions
    # hm_nde_test_SMHR_q0, hm_nde_test_SMHR_q1, hm_nde_test_SMHR_q2 = np.quantile(hm_train_SMHR_test, (0.16, 0.5, 0.84), axis=1)

    return y_test, y_nde_test_mags, hm_train_SMHR_test
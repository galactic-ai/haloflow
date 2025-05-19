from haloflow import data as D
from haloflow import util as U
from haloflow.config import get_dat_dir
from haloflow.dann.evalutate import evaluate
from haloflow.dann.model import DANNModel
from haloflow import corr as Corr

import torch
import numpy as np
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
        n_samples=None,
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
            qphi.sample((samples // len(qphis),), x=X_test_torch[i], show_progress_bars=False)
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


def plot_true_pred(ax, 
                   train_obs, 
                   dann_sim,
                   npe_train_sim, 
                   test_obs, 
                   test_sim, 
                   device,
                   fmt='.C0',
                   mass='halo',
                   use_weights=False,
                   **valid_kwargs):
    """
    Plotting script for true vs predicted values.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on
    train_obs : str
        Observational data to train on. 
        For now, it's either 'mags' or 'mags_morph'
    train_sim : str
        Simulation data to train on. 
        For now, it's either 'TNG100', 'Eagle100', 'TNG50', 'TNG_ALL', 'Simba100'
    test_obs : str
        Observational data to test on. 
        For now, it's either 'mags' or 'mags_morph'
    test_sim : str
        Simulation data to test on. 
        For now, it's either 'TNG100', 'Eagle100', 'TNG50', 'TNG_ALL', 'Simba100'
    device : str
        Device to run on. 
        For now, it's either 'cpu' or 'cuda'
    mass : str
        Mass to predict. 
        For now, it's either 'halo' or 'halo'
    use_weights : bool
        Whether to use weights to correct for SMF and HMF implicit prior.
    
    Returns
    -------
    ax : matplotlib axis
    """
    Y_test, _ = D.hf2_centrals('test', test_obs, test_sim)

    # randomly choose 100 galaxies
    np.random.seed(42)
    idx = np.random.choice(len(Y_test), 100, replace=False)
    y_true = Y_test[idx]

    _, _, _, y_nde = validate_npe(train_obs, 
                                  dann_sim, 
                                  npe_train_sim, 
                                  test_obs, 
                                  test_sim, 
                                  device=device, 
                                  train_samples=100, 
                                  n_samples=1000, 
                                  **valid_kwargs,
                                  )

    if use_weights:
        # apply weights to correct for SMF and HMF implicit prior
        # Initialize lists to store the resampled M* and Mh values
        y_nde_resampled_Ms = []
        y_nde_resampled_Mh = []

        # Loop over each sample (i) in the second axis (n_samples)
        for i in range(y_nde.shape[0]):
            # Extract the i-th slice of y_nde (M* and Mh values for this sample)
            y_sample = y_nde[i, :, :]
            
            # Compute the weights for the M* and Mh prior for this sample
            w_smf, w_hmf = Corr.w_prior_corr(Y_sam=y_sample, sim=test_sim, bins=10, version=1)
            
            # Resample M* using w_smf
            resampled_Ms = Corr.weighted_resample(y_sample[:, 0], w_smf)
            
            # Resample Mh using w_hmf
            resampled_Mh = Corr.weighted_resample(y_sample[:, 0], w_hmf)
            
            # Append the resampled M* and Mh values to the lists
            y_nde_resampled_Ms.append(resampled_Ms)
            y_nde_resampled_Mh.append(resampled_Mh)

        # Convert the lists to numpy arrays and combine them into a final array
        y_nde_resampled_Ms = np.array(y_nde_resampled_Ms)  # Shape: (100, 1000)
        y_nde_resampled_Mh = np.array(y_nde_resampled_Mh)  # Shape: (100, 1000)

        # Stack the resampled M* and Mh values to get the final resampled array
        y_nde = np.stack([y_nde_resampled_Ms, y_nde_resampled_Mh], axis=-1)  # Shape: (100, 1000, 2)

    y_nde_q0, y_nde_q1, y_nde_q2 = np.quantile(y_nde, (0.16, 0.5, 0.84), axis=1)
    ax.plot([9.5, 14.], [9.5, 14.], c='k', ls='--', label='_nolegend_')

    # ax.text(0.05, 0.95, f'{train_sim.upper()}-{test_sim.upper()}', transform=ax.transAxes, ha='left', va='top', fontsize=20)
    if mass == 'stellar':
        ax.errorbar(y_true[:,0], y_nde_q1[:,0], 
                    yerr=[y_nde_q1[:,0] - y_nde_q0[:,0], y_nde_q2[:,0] - y_nde_q1[:,0]], 
                    fmt=fmt, label=f'{npe_train_sim.upper()}-{test_sim.upper()}')

        ax.set_xlabel(r"$\log M_*$ (true)", fontsize=25)
        ax.set_ylabel(r"$\log M_*$ (predicted)", fontsize=25)

        ax.set_xlim(9.5, 12.)
        ax.set_ylim(9.5, 12.)
    
    elif mass == 'halo':
        ax.errorbar(y_true[:,1], y_nde_q1[:,1], 
                    yerr=[y_nde_q1[:,1] - y_nde_q0[:,1], y_nde_q2[:,1] - y_nde_q1[:,1]], 
                    fmt=fmt, label=f'{npe_train_sim.upper()}-{test_sim.upper()}')

        ax.set_xlabel(r"$\log M_h$ (true)", fontsize=25)
        ax.set_ylabel(r"$\log M_h$ (predicted)", fontsize=25)

        ax.set_xlim(10., 14.)
        ax.set_ylim(10., 14.)
    else:
        raise ValueError(f"mass should be either 'halo' or 'halo', but got {mass}")

    return ax, y_true, y_nde


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_obs = 'mags_morph_extra'
    dann_sim = 'Eagle100'
    npe_train_sim = 'TNG_ALL'
    samples = 1_000

    ranks, alpha, ecp, y_nde = validate_npe(
        npe_train_obs='mags_morph_extra',
        dann_sim=dann_sim,
        npe_train_sim=npe_train_sim,
        test_obs=test_obs,
        test_sim=npe_train_sim,
        device=device,
        data_dir = get_dat_dir() + 'hf2/npe/',
        n_ensemble=5,
        n_samples=samples,
        train_samples=None,
        version=1,
        with_dann=True,
    )

    ranks_2, alpha_2, ecp_2, y_nde_2 = validate_npe(
        npe_train_obs='mags_morph_extra',
        dann_sim=dann_sim,
        npe_train_sim=npe_train_sim,
        test_obs=test_obs,
        test_sim=dann_sim,
        device=device,
        data_dir = get_dat_dir() + 'hf2/npe/',
        n_ensemble=5,
        n_samples=samples,
        train_samples=None,
        version=1,
        with_dann=True,
    )

    # plotting
    import matplotlib.pyplot as plt
    from haloflow.npe.plotting import plot_rank_statistics, plot_coverage

    fig, ax = plot_coverage(
        [alpha, alpha_2],
        [ecp, ecp_2],
        labels=[f'ND {npe_train_sim}-{npe_train_sim}', f'ND {npe_train_sim}-{dann_sim}'],
    )
    plt.show()


    # fig, ax = plt.subplots(2, 1, figsize=(3, 5), dpi=150, sharex=True)

    # ax[0], y_true, y_pred = plot_true_pred(
    #     ax[0],
    #     train_obs='mags_morph_extra',
    #     dann_sim=dann_sim,
    #     npe_train_sim=npe_train_sim,
    #     test_obs=test_obs,
    #     test_sim=npe_train_sim,
    #     device='cpu',
    #     mass='stellar',
    #     data_dir=get_dat_dir() + 'hf2/npe/',
    #     with_dann=False
    # )

    # ax[0], y_true_eg, y_pred_eg = plot_true_pred(
    #     ax[0],
    #     train_obs='mags_morph_extra',
    #     dann_sim=dann_sim,
    #     npe_train_sim=npe_train_sim,
    #     test_obs=test_obs,
    #     test_sim=dann_sim,
    #     device='cpu',
    #     mass='stellar',
    #     data_dir=get_dat_dir() + 'hf2/npe/',
    #     fmt='.C1',
    #     use_weights=True,
    #     with_dann=False
    # )

    # ax[0], y_true_eg_2, y_pred_eg_2 = plot_true_pred(
    #     ax[0],
    #     train_obs='mags_morph_extra',
    #     dann_sim=dann_sim,
    #     npe_train_sim=npe_train_sim,
    #     test_obs=test_obs,
    #     test_sim=npe_train_sim,
    #     device='cpu',
    #     mass='stellar',
    #     data_dir=get_dat_dir() + 'hf2/npe/',
    #     fmt='.C2',
    #     with_dann=True
    # )

    # ax[0], y_true_eg_3, y_pred_eg_3 = plot_true_pred(
    #     ax[0],
    #     train_obs='mags_morph_extra',
    #     dann_sim=dann_sim,
    #     npe_train_sim=npe_train_sim,
    #     test_obs=test_obs,
    #     test_sim=dann_sim,
    #     device='cpu',
    #     mass='stellar',
    #     data_dir=get_dat_dir() + 'hf2/npe/',
    #     fmt='.C3',
    #     use_weights=True,
    #     with_dann=True
    # )

    # ax[0].legend([f'N {npe_train_sim.upper()}-{npe_train_sim.upper()}',
    #                f'N {npe_train_sim.upper()}-{dann_sim.upper()}',
    #                f'ND {npe_train_sim.upper()}-{npe_train_sim.upper()}',
    #                f'ND {npe_train_sim.upper()}-{dann_sim.upper()}'],)

    # y_nde_q0, y_nde_q1, y_nde_q2 = np.quantile(y_pred, (0.16, 0.5, 0.84), axis=1)
    # y_nde_eg_q0, y_nde_eg_q1, y_nde_eg_q2 = np.quantile(y_pred_eg, (0.16, 0.5, 0.84), axis=1)
    # y_nde_dann_q0, y_nde_dann_q1, y_nde_dann_q2 = np.quantile(y_pred_eg_2, (0.16, 0.5, 0.84), axis=1)
    # y_nde_eg_3_q0, y_nde_eg_3_q1, y_nde_eg_3_q2 = np.quantile(y_pred_eg_3, (0.16, 0.5, 0.84), axis=1)


    # ax[1].plot(y_true[:, 0], y_true[:, 0] - y_nde_q1[:, 0], '.C0', label='')
    # ax[1].plot(y_true_eg[:, 0], y_true_eg[:, 0] - y_nde_eg_q1[:, 0], '.C1', label='')
    # ax[1].plot(y_true[:, 0], y_true[:, 0] - y_nde_dann_q1[:, 0], '.C2', label='')
    # ax[1].plot(y_true_eg_3[:, 0], y_true_eg_3[:, 0] - y_nde_eg_3_q1[:, 0], '.C3', label='')

    # # # dashed lines at 0
    # ax[1].axhline(0, color='black', linestyle='--')
    # ax[1].set_xlabel('True', fontsize=25)
    # ax[1].set_ylabel('True - Predicted', fontsize=25)

    # plt.show()




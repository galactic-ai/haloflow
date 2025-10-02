import matplotlib.pyplot as plt
import numpy as np

from .. import config as C
from . import valid as V
from .. import data as D
from .. import corr as Corr

C.setup_plotting_config()

def plot_rank_statistics(ranks_list, labels):
    """
    Example: 
    ranks_list = [ranks_nde, ranks_nde_2]
    labels = ['NDE', 'NDE 2']
    plot_rank_statistics(ranks_list, labels)
    """
    num_ranks = ranks_list[0].shape[1]
    fig, axes = plt.subplots(1, num_ranks, figsize=(6 * num_ranks, 3.5), dpi=150)

    if num_ranks == 1:
        axes = [axes]

    for i in range(num_ranks):
        for ranks, label in zip(ranks_list, labels):
            axes[i].hist(ranks[:, i], range=(0., 1), bins=20, histtype='step', density=True, linewidth=2, label=label)
        axes[i].plot([0., 1.], [1., 1.], c='k', ls='--')
        axes[i].set_xlabel('rank statistics', fontsize=20)
        axes[i].set_xlim(0., 1.)
        axes[i].set_ylim(0., 3.)
        axes[i].set_yticks([])
        # axes[i].text(0.05, 0.95, f'Rank {i+1}', fontsize=20, transform=axes[i].transAxes, ha='left', va='top')
    
    axes[0].legend()

    plt.tight_layout()
    return fig

def plot_coverage(alpha_list, ecp_list, labels, ax=None):
    """
    Example:
    alpha_list = [alpha_nde, alpha_nde_2]
    ecp_list = [ecp_nde, ecp_nde_2]
    labels = ['NDE', 'NDE 2']
    plot_coverage(alpha_list, ecp_list, labels)
    """
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    else:
        fig = ax.get_figure()
        
    ax.plot([0, 1], [0, 1], ls="--", color="k")
    
    for alpha, ecp, label in zip(alpha_list, ecp_list, labels):
        ax.plot(alpha, ecp, label=label)
    
    ax.legend(loc='lower right', fontsize=15)
    ax.set_ylabel("Expected Coverage", fontsize=20)
    ax.set_ylim(0., 1.)
    ax.set_xlabel("Credibility Level", fontsize=20)
    ax.set_xlim(0., 1.)
    
    return fig, ax

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
                   train_samples=100,
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

    np.random.seed(42)
    # instance is a number of samples then do random sampling
    if isinstance(train_samples, int):
        # randomly choose 100 galaxies
        idx = np.random.choice(len(Y_test), train_samples, replace=False)
    # instance is a list or numpy array of indices
    elif isinstance(train_samples, (list, np.ndarray)):
        idx = train_samples
    else:
        raise ValueError("train_samples must be an int, list, or np.ndarray")
    y_true = Y_test[idx]

    _, _, _, y_nde = V.validate_npe(train_obs, 
                                  dann_sim, 
                                  npe_train_sim, 
                                  test_obs, 
                                  test_sim, 
                                  device=device, 
                                  n_samples=1000, 
                                  train_samples=train_samples,
                                  **valid_kwargs,
                                  )

    if mass == 'stellar':
        indx = 0
    elif mass == 'halo':
        indx = 1
    else:
        raise ValueError(f"mass should be either 'stellar' or 'halo', but got {mass}")

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
            resampled_Mh = Corr.weighted_resample(y_sample[:, 1], w_hmf)
            
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
    ax.errorbar(y_true[:,indx], y_nde_q1[:,indx], 
                yerr=[y_nde_q1[:,indx] - y_nde_q0[:,indx], y_nde_q2[:,indx] - y_nde_q1[:,indx]], 
                fmt=fmt, ms=8, label=f'{npe_train_sim.upper()}-{test_sim.upper()}', alpha=0.7)

    if mass == 'stellar':
        ax.set_xlabel(r"true $\log M_*$", fontsize=20)
        ax.set_ylabel(r"inferred $\log M_*$", fontsize=20)

        ax.set_xlim(9.5, 12.)
        ax.set_ylim(9.5, 12.)
    
    elif mass == 'halo':
        ax.set_xlabel(r"true $\log M_{\rm h}$", fontsize=20)
        ax.set_ylabel(r"inferred $\log M_{\rm h}$", fontsize=20)

        ax.set_xlim(11.3, 14.)
        ax.set_ylim(11.3, 14.)
    else:
        raise ValueError(f"mass should be either 'halo' or 'halo', but got {mass}")

    return ax, y_true, y_nde


def get_sample_indices(dataset, obs, sim, max_per_bin=15):
    """Get sample indices for a given dataset to be used for plotting.

    Parameters
    ----------

    dataset : str
        Specify training or test data. Specify 'train' or 'test' or 'all.
    obs : str
        Specify the observables to include. ('mags', 'mags_morph', 'mags_morph_extra')
    sim : str
        Specify the simulation to use. ('TNG100', 'Eagle100', 'TNG50', 'TNG_ALL', 'Simba100')
    max_per_bin : int
        Specify the maximum number of samples to include per bin.
    
    Returns
    -------
    selected_indices : np.ndarray
        Array of selected indices for the specified dataset.
    """

    # set random seed
    np.random.seed(42)

    y, _ = D.hf2_centrals(dataset, obs, sim)
    halo_masses = y[:, 1]

    hm_bins = np.linspace(11, 15., 10)

    bin_indices = np.digitize(halo_masses, hm_bins)

    selected_indices = []
    for i in range(1, len(hm_bins)):
        indices_in_bin = np.where(bin_indices == i)[0]
        if len(indices_in_bin) > max_per_bin:
            selected_indices.extend(np.random.choice(indices_in_bin, max_per_bin, replace=False))
        else:
            selected_indices.extend(indices_in_bin.tolist())
    selected_indices = np.array(selected_indices)

    return selected_indices
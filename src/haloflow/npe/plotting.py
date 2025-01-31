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
        axes[i].text(0.05, 0.95, f'Rank {i+1}', fontsize=20, transform=axes[i].transAxes, ha='left', va='top')
    
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

def plot_true_pred(ax, train_obs, train_sim, 
                   test_obs, test_sim, device,
                   fmt='.C0',
                   mass='halo',
                   use_weights=False):
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
        For now, it's either 'halo' or 'stellar'
    use_weights : bool
        Whether to use weights to correct for SMF and HMF implicit prior.
    
    Returns
    -------
    ax : matplotlib axis
    """
    Y_test, _ = D.hf2_centrals('test', test_obs, test_sim)

    # randomly choose 100 galaxies
    idx = np.random.choice(len(Y_test), 100, replace=False)
    y_true = Y_test[idx]

    _, _, _, y_nde = V.validate_npe(train_obs, train_sim, test_obs, test_sim, device=device, train_samples=100, n_samples=1000)

    if use_weights:
        # apply weights to correct for SMF and HMF implicit prior
        w_smf, w_hmf = Corr.w_prior_corr(y_nde, train_sim, bins=10, version=1)
        y_nde[:, 0] = Corr.weighted_resample(y_nde[:, 0], w_smf)
        y_nde[:, 1] = Corr.weighted_resample(y_nde[:, 1], w_hmf)

    y_nde_q0, y_nde_q1, y_nde_q2 = np.quantile(y_nde, (0.16, 0.5, 0.84), axis=1)
    ax.plot([9.5, 12.], [9.5, 12.], c='k', ls='--')

    # ax.text(0.05, 0.95, f'{train_sim.upper()}-{test_sim.upper()}', transform=ax.transAxes, ha='left', va='top', fontsize=20)
    if mass == 'stellar':
        ax.errorbar(y_true[:,0], y_nde_q1[:,0], 
                    yerr=[y_nde_q1[:,0] - y_nde_q0[:,0], y_nde_q2[:,0] - y_nde_q1[:,0]], 
                    fmt=fmt, label=f'{train_sim.upper()}-{test_sim.upper()}')

        ax.set_xlabel(r"$\log M_*$ (true)", fontsize=25)
        ax.set_ylabel(r"$\log M_*$ (predicted)", fontsize=25)
    
    elif mass == 'halo':
        ax.errorbar(y_true[:,1], y_nde_q1[:,1], 
                    yerr=[y_nde_q1[:,1] - y_nde_q0[:,1], y_nde_q2[:,1] - y_nde_q1[:,1]], 
                    fmt=fmt, label=f'{train_sim.upper()}-{test_sim.upper()}')

        ax.set_xlabel(r"$\log M_h$ (true)", fontsize=25)
        ax.set_ylabel(r"$\log M_h$ (predicted)", fontsize=25)
    else:
        raise ValueError(f"mass should be either 'halo' or 'stellar', but got {mass}")

    ax.set_xlim(9.5, 12.)
    ax.set_ylim(9.5, 12.)

    return ax

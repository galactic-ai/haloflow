import matplotlib.pyplot as plt

from .. import config as C

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

def plot_coverage(alpha_list, ecp_list, labels):
    """
    Example:
    alpha_list = [alpha_nde, alpha_nde_2]
    ecp_list = [ecp_nde, ecp_nde_2]
    labels = ['NDE', 'NDE 2']
    plot_coverage(alpha_list, ecp_list, labels)
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    ax.plot([0, 1], [0, 1], ls="--", color="k")
    
    for alpha, ecp, label in zip(alpha_list, ecp_list, labels):
        ax.plot(alpha, ecp, label=label)
    
    ax.legend(loc='lower right', fontsize=15)
    ax.set_ylabel("Expected Coverage", fontsize=20)
    ax.set_ylim(0., 1.)
    ax.set_xlabel("Credibility Level", fontsize=20)
    ax.set_xlim(0., 1.)
    
    return fig, ax
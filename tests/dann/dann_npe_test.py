import torch
import numpy as np

from haloflow.npe import valid as V
from haloflow.npe.plotting import plot_true_pred
from haloflow.config import get_dat_dir


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_obs = 'mags_morph_extra'
    dann_sim = 'Eagle100'
    npe_train_sim = 'Simba100'
    samples = 1_000

    ranks, alpha, ecp, y_nde = V.validate_npe(
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

    ranks_2, alpha_2, ecp_2, y_nde_2 = V.validate_npe(
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
    plt.savefig(f'rank_m{dann_sim}_{npe_train_sim}_{test_obs}.png', bbox_inches='tight')
    plt.clf()


    fig, ax = plt.subplots(2, 1, figsize=(6, 10), dpi=150, sharex=True)
    ax = ax.flatten()

    ax[0], y_true, y_pred = plot_true_pred(
        ax[0],
        train_obs='mags_morph_extra',
        dann_sim=dann_sim,
        npe_train_sim=npe_train_sim,
        test_obs=test_obs,
        test_sim=npe_train_sim,
        device='cpu',
        mass='halo',
        data_dir=get_dat_dir() + 'hf2/npe/',
        with_dann=False
    )

    ax[0], y_true_eg, y_pred_eg = plot_true_pred(
        ax[0],
        train_obs='mags_morph_extra',
        dann_sim=dann_sim,
        npe_train_sim=npe_train_sim,
        test_obs=test_obs,
        test_sim=dann_sim,
        device='cpu',
        mass='halo',
        data_dir=get_dat_dir() + 'hf2/npe/',
        fmt='.C1',
        use_weights=True,
        with_dann=False
    )

    ax[0], y_true_eg_2, y_pred_eg_2 = plot_true_pred(
        ax[0],
        train_obs='mags_morph_extra',
        dann_sim=dann_sim,
        npe_train_sim=npe_train_sim,
        test_obs=test_obs,
        test_sim=npe_train_sim,
        device='cpu',
        mass='halo',
        data_dir=get_dat_dir() + 'hf2/npe/',
        fmt='.C2',
        with_dann=True
    )

    ax[0], y_true_eg_3, y_pred_eg_3 = plot_true_pred(
        ax[0],
        train_obs='mags_morph_extra',
        dann_sim=dann_sim,
        npe_train_sim=npe_train_sim,
        test_obs=test_obs,
        test_sim=dann_sim,
        device='cpu',
        mass='halo',
        data_dir=get_dat_dir() + 'hf2/npe/',
        fmt='.C3',
        use_weights=True,
        with_dann=True
    )

    ax[0].legend([f'N {npe_train_sim.upper()}-{npe_train_sim.upper()}',
                   f'N {npe_train_sim.upper()}-{dann_sim.upper()}',
                   f'ND {npe_train_sim.upper()}-{npe_train_sim.upper()}',
                   f'ND {npe_train_sim.upper()}-{dann_sim.upper()}'],)
    

    y_nde_q0, y_nde_q1, y_nde_q2 = np.quantile(y_pred, (0.16, 0.5, 0.84), axis=1)
    y_nde_eg_q0, y_nde_eg_q1, y_nde_eg_q2 = np.quantile(y_pred_eg, (0.16, 0.5, 0.84), axis=1)
    y_nde_dann_q0, y_nde_dann_q1, y_nde_dann_q2 = np.quantile(y_pred_eg_2, (0.16, 0.5, 0.84), axis=1)
    y_nde_eg_3_q0, y_nde_eg_3_q1, y_nde_eg_3_q2 = np.quantile(y_pred_eg_3, (0.16, 0.5, 0.84), axis=1)


    ax[1].plot(y_true[:, 1], y_true[:, 1] - y_nde_q1[:, 1], '.C0', label='')
    ax[1].plot(y_true_eg[:, 1], y_true_eg[:, 1] - y_nde_eg_q1[:, 1], '.C1', label='')
    ax[1].plot(y_true[:, 1], y_true[:, 1] - y_nde_dann_q1[:, 1], '.C2', label='')
    ax[1].plot(y_true_eg_3[:, 1], y_true_eg_3[:, 1] - y_nde_eg_3_q1[:, 1], '.C3', label='')

    # dashed lines at 0
    ax[1].axhline(0, color='black', linestyle='--')
    ax[1].set_xlabel('True', fontsize=25)
    ax[1].set_ylabel('True - Predicted', fontsize=25)

    plt.savefig(f'true_pred_m{dann_sim}_{npe_train_sim}_{test_obs}_halo.png', bbox_inches='tight')
    plt.clf()


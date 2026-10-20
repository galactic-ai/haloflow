# halo**flow**
[![HaloFlow I](https://img.shields.io/badge/arXiv-2310.04503-b31b1b.svg)](https://arxiv.org/abs/2310.04503)
[![HaloFlow II](https://img.shields.io/badge/arXiv-2603.12380-b31b1b.svg)](https://arxiv.org/abs/2603.12380)
[![HaloFlow II code](https://img.shields.io/badge/code_v2.0.0-10.5281%2Fzenodo.23497593-blue.svg)](https://doi.org/10.5281/zenodo.23497593)
[![HaloFlow II data](https://img.shields.io/badge/data-10.5281%2Fzenodo.23497672-blue.svg)](https://doi.org/10.5281/zenodo.23497672)

`haloflow` is a simulation-based inference (SBI) framework for inferring the host
dark matter halo mass (M<sub>h</sub>), jointly with stellar mass (M<sub>*</sub>), of central
galaxies from their photometry and morphology. It uses neural posterior estimation (NPE)
with normalizing flows trained on forward-modeled, HSC-SSP-like synthetic images of galaxies from
cosmological hydrodynamical simulations.

**HaloFlow II** adds domain adaptation (DA). A DA network — a Domain-Adversarial Neural
Network (DANN) or a Maximum Mean Discrepancy (MMD) model — compresses the galaxy features
into a representation aligned across simulations. The NPE is then trained on that
representation, which makes the M<sub>h</sub> posteriors more robust when the model is applied to galaxies
from a different simulation (and, ultimately, to observations).

## Versions

Each paper corresponds to a tagged version of this repository:

| Tag | Paper | Notes |
|---|---|---|
| `v1.0.0` | HaloFlow I (Hahn, Bottrell & Lee 2024, ApJ 968, 90) | NPE on TNG images; photometry + morphology + satellite features |
| `v2.0.0-alpha.1` | Robust Halo Masses using HaloFlow with Domain Adaptation (NeurIPS 2025 ML4PS workshop) | first DANN/MMD domain-adaptation results |
| `v2.0.0-rc.1` | HaloFlow II as submitted to ApJ (arXiv:2603.12380) | DA across TNG, EAGLE and SIMBA |
| `v2.0.0` | HaloFlow II, revised version (ApJ resubmission) | revised after the referee report; archived on Zenodo |

To check out the code for a given paper: `git checkout <tag>`.

## Installation

```bash
git clone https://github.com/galactic-ai/haloflow.git
cd haloflow
python -m venv venv && source venv/bin/activate   # Python 3.13 was used for HaloFlow II
pip install -r requirements.txt
pip install -e .
```

`haloflow.config.get_dat_dir()` returns the data directory. Unless the environment variable
`machine` is set to a known cluster (see `src/haloflow/config.py`), it falls back to the relative
path `../../data/`, so run scripts from a subdirectory of `bin/` (e.g. `cd bin/npe`).

On recent macOS versions, `scipy==1.15.2` (pinned in `requirements.txt`) fails to load; install
`scipy>=1.15.3` (HaloFlow II was finalised with 1.18.1).

## Repository layout

```
src/haloflow/
    data.py         data loading, selection cuts and train/test split (hf2_centrals)
    corr.py         implicit-prior (SMF/HMF) correction of NPE posteriors
    schechter.py    Schechter-function fits used for prior correction and sample weights
    util.py         NPE ensemble loading, metrics (normalized residual beta), helpers
    npe/            NPE training (Optuna) and validation, SHMR baseline, plotting
    dann/           DANN model, evaluation and visualisation
    mmd/            MMD model and prediction helpers
bin/
    data/           preprocessing of the simulation catalogs (incl. TNG50 -> TNG100 correction)
    npe/            baseline NPE training / validation
    dann/           DANN training and DANN+NPE training
    mmd/            MMD training and MMD+NPE training
    plotting/       UMAP and summary plots
    hf2_paper/      HaloFlow II tables and figures (from the Zenodo release) and the scripts that made its data
nb/                 exploratory notebooks (HaloFlow I and early HaloFlow II)
tests/              consistency checks and scatter-plot tests
```

## Data

The inputs are feature catalogs (`hf2.<sim>.morph_subhalo.csv`) measured with ProFuse from
synthetic HSC images of central galaxies at z = 0.1 in TNG50, TNG100, EAGLE and SIMBA
(Bottrell et al. 2024). The synthetic images are public (TNG: the
[IllustrisTNG website](https://www.tng-project.org/explore/gallery/bottrell23/)); the feature
catalogs themselves are not distributed with this repository.
<!-- TODO: link for the EAGLE/SIMBA images (ask Connor) -->

The trained NPE ensembles and DA networks, the posterior samples for all test galaxies, and
machine-readable versions of the paper's tables are archived on Zenodo ([10.5281/zenodo.23497672](https://doi.org/10.5281/zenodo.23497672)).

## Reproducing HaloFlow II

The tables and most figures can be remade from the Zenodo release alone with the scripts in
`bin/hf2_paper/`. They read the archive from `--zenodo <path>` or the environment variable
`HALOFLOW_ZENODO`, and need only `numpy`, `matplotlib` (with LaTeX) and `haloflow.config`.
Training, and drawing new posterior samples, need the feature catalogs (see Data).

| Paper item | Script | Needs |
|---|---|---|
| Train DANN / MMD (per target simulation) | `bin/dann/train.py --sim <target>`, `bin/mmd/train.py --sim <target>` | catalogs |
| Train NPEs on DA features | `bin/dann/npe_dann.py`, `bin/mmd/npe_mmd.py` | catalogs |
| Train baseline NPEs | `bin/npe/npe.py` | catalogs |
| Posterior samples (Zenodo `posteriors/samples`) | `bin/hf2_paper/get_bias_estimates.py` | catalogs + models |
| DRP coverage curves (Zenodo `coverage/`) | `bin/hf2_paper/drp_coverage_run.py` | catalogs + models |
| Standard-method baseline (Zenodo `posteriors/standard_method/`) | `bin/hf2_paper/standard_method_run.py` | catalogs + models |
| Tables 2, 3, 7, 8 | `bin/hf2_paper/tables_beta.py` | Zenodo |
| Bootstrap uncertainties (Sec. 4.2) | `bin/hf2_paper/bootstrap_beta.py` | Zenodo |
| Fig. 3 (UMAP) | `bin/plotting/get_umap_vis.ipynb` | catalogs + DA networks |
| Fig. 4 | `bin/hf2_paper/fig4_model_mismatch.py` | Zenodo |
| Fig. 5 | `bin/hf2_paper/fig5_scatter_grid.py` | Zenodo |
| Fig. 6 | `bin/hf2_paper/fig6_precision_bias.py` | Zenodo |
| Fig. 7 | `bin/npe/smhr.ipynb` (cells 19, 24) | catalogs + models |
| Fig. 8 | `bin/hf2_paper/fig8_drp_coverage.py` | Zenodo |

```bash
cd bin/hf2_paper
export HALOFLOW_ZENODO=/path/to/zenodo/archive
python tables_beta.py                     # prints Tables 2, 3, 7, 8 and checks them against the archive
python fig5_scatter_grid.py --out figures/
```

Figures made from the Zenodo files use the archived posterior draws. The published Figs. 4–7
used separate draws from the same models, so the points differ slightly (see the Zenodo README).

## Publications

- **HaloFlow. I. Neural Inference of Halo Mass from Galaxy Photometry and Morphology**<br>
  ChangHoon Hahn, Connor Bottrell, Khee-Gan Lee<br>
  ApJ 968, 90 (2024) · [arXiv:2310.04503](https://arxiv.org/abs/2310.04503)
- **Robust Halo Masses using HaloFlow with Domain Adaptation**<br>
  Nikhil Garuda, Khee-Gan Lee, ChangHoon Hahn, Connor Bottrell<br>
  NeurIPS 2025 Workshop on Machine Learning and the Physical Sciences (ML4PS), published 7 Oct 2025
- **HaloFlow II: Robust Galaxy–Halo Mass Inference with Domain Adaptation**<br>
  Nikhil Garuda, ChangHoon Hahn, Connor Bottrell, Khee-Gan Lee<br>
  submitted to ApJ · [arXiv:2603.12380](https://arxiv.org/abs/2603.12380)

If you use this code, please cite the relevant paper(s) above and the Zenodo archive of the
version you used.

## Team

[Nikhil Garuda](https://github.com/Nikhil0504) (UT Austin)

[ChangHoon Hahn](https://github.com/changhoonhahn) (UT Austin)

[Connor Bottrell](https://github.com/cbottrell) (ICRAR, University of Western Australia)

[Khee-Gan Lee](https://sites.google.com/view/kglee/home) (Kavli IPMU)

## License

MIT (see `LICENSE`).

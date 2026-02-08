'''

module to correct for implicity prior imposed on M* 
and Mh from the SMF and HMF. 

'''
import numpy as np 
import warnings 
from functools import partial
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from . import data as D 
from . import util as U
from . import schechter as S
from scipy.optimize import minimize

def resample_one(y_sample, test_sim):
    w_smf, w_hmf = w_prior_corr(
        Y_sam=y_sample, sim=test_sim, bins=10, version=1
    )
    return (
        weighted_resample(y_sample[:, 0], w_smf),
        weighted_resample(y_sample[:, 1], w_hmf),
    )


def w_prior_corr(Y_sam, sim, bins=10, version=1): 
    ''' return weights to correct for SMF and HMF implicit prior for 
    **marginalized 1d posterior**. 

    args: 
        Y_sam (Nx2 array): sample of M*, Mh drawn from some posterior p(M*, Mh|X) 

        sim (str): specify the simulation used 

    kwargs: 
        bins (int): specify the number of bins for the SMF and HMF used in correction

        version (int): specify version of dataset 

    return: 
        w_smf (N array): weights for correcting the SMF  priors respectively. 

        w_hmf (N array): weights for correcting the HMF  priors respectively. 
    '''
    # get M* and Mh of entire simulation 
    Y_all, _ = D.hf2_centrals('all', 'mags', sim=sim, version=version) 
    mask = (Y_all[:, 1] >= 12)
    Y_all = Y_all[mask]

    # check ranges of sampled M* and Mh 
    if (Y_sam[:,0].min() < Y_all[:,0].min()) or (Y_sam[:,0].max() > Y_all[:,0].max()):
        warnings.warn('some M* samples are below or above the M* limit')
    if (Y_sam[:,1].min() < Y_all[:,1].min()) or (Y_sam[:,1].max() > Y_all[:,1].max()):
        warnings.warn('some M* samples are below or above the Mh limit')

    # implicit p(M*) prior
    p_smf, ms_bin = np.histogram(Y_all[:,0], bins=bins) 
    lo_ms, hi_ms = ms_bin[0], ms_bin[-1]
    nll_smf_fit = partial(S.nll_counts, p_hmf=p_smf, mh_bin=ms_bin, lo=lo_ms, hi=hi_ms, type='single_schechter')
    p0 = [np.log(0.6*p_smf.sum()+1e-6), np.median(Y_all[:,0]), -1.0]
    bounds = [(-50, 50), (lo_ms, hi_ms), (-5, 2)]
    res = minimize(nll_smf_fit, p0, bounds=bounds)
    mu_bin_ms = S.mu_counts(res.x, type='single_schechter', mh_bin=ms_bin, lo=lo_ms, hi=hi_ms)

    # implicit p(Mh) prior
    p_hmf, mh_bin = np.histogram(Y_all[:,1], bins=bins) 
    lo_mh, hi_mh = mh_bin[0], mh_bin[-1]
    nll_mh_fit = partial(S.nll_counts, p_hmf=p_hmf, mh_bin=mh_bin, lo=lo_mh, hi=hi_mh, type='single_schechter')
    p0 = [np.log(0.6*p_hmf.sum()+1e-6), np.median(Y_all[:,1]), -1.0]
    bounds = [(-50, 50), (lo_mh, hi_mh), (-5, 2)]
    res = minimize(nll_mh_fit, p0, bounds=bounds)
    mu_bin_mh = S.mu_counts(res.x, type='single_schechter', mh_bin=mh_bin, lo=lo_mh, hi=hi_mh)
    
    # weights for M* prior 
    ms_sam = (Y_sam[:,0].copy()).clip(lo_ms, hi_ms) # clip the edges 
    idx = np.digitize(ms_sam, ms_bin) - 1
    idx = np.clip(idx, 0, len(mu_bin_ms)-1)
    # w_smf = 1./np.interp(ms_sam, 0.5 * (ms_bin[:-1] + ms_bin[1:]), p_smf) 
    w_smf = 1./mu_bin_ms[idx]
    w_smf *= float(len(w_smf)) / np.sum(w_smf) # renormalize (this is for convenience and should not affect anything) 

    # weights for Mh prior 
    mh_sam = (Y_sam[:,1].copy()).clip(lo_mh, hi_mh) # clip the edges 
    idx = np.digitize(mh_sam, mh_bin) - 1
    idx = np.clip(idx, 0, len(mu_bin_mh)-1)
    # w_hmf = 1. / np.interp(mh_sam, 0.5 * (mh_bin[:-1] + mh_bin[1:]), p_hmf)
    w_hmf = 1. / mu_bin_mh[idx]
    w_hmf *= float(len(w_hmf)) / np.sum(w_hmf) # renormalize (this is for convenience and should not affect anything) 

    return w_smf, w_hmf

def weighted_resample(data, weights):
    """
    Returns an approximate resampled array based on float weights.
    data: 1D array of values
    weights: corresponding 1D array of float weights
    """
    # Normalize weights so they sum to 1
    normalized_weights = weights / np.sum(weights)
    
    # Number of points to sample can be decided, e.g. same length as original
    n_samples = len(data)
    
    # Choose indices by probability
    np.random.seed(42)
    indices = np.random.choice(np.arange(len(data)), size=n_samples, p=normalized_weights)
    
    # Return the chosen elements
    return data[indices]

def weight_nde(y_nde, test_sim):
    with U.tqdm_joblib(tqdm(total=y_nde.shape[0], desc="Resampling")):
        results = Parallel(n_jobs=8)(
            delayed(resample_one)(y_nde[i], test_sim) for i in range(y_nde.shape[0])
        )

    y_nde_resampled_Ms, y_nde_resampled_Mh = map(np.array, zip(*results))
    # Stack the resampled M* and Mh values to get the final resampled array
    y_nde_resample = np.stack([y_nde_resampled_Ms, y_nde_resampled_Mh], axis=-1)  # Shape: (100, 1000, 2)

    return y_nde_resample


# def weight_nde(y_nde, test_sim):
#     # apply weights to correct for SMF and HMF implicit prior
#     # Initialize lists to store the resampled M* and Mh values
#     y_nde_resampled_Ms = []
#     y_nde_resampled_Mh = []

#     # Loop over each sample (i) in the second axis (n_samples)
#     for i in range(y_nde.shape[0]):
#         # Extract the i-th slice of y_nde (M* and Mh values for this sample)
#         y_sample = y_nde[i, :, :]
        
#         # Compute the weights for the M* and Mh prior for this sample
#         w_smf, w_hmf = Corr.w_prior_corr(Y_sam=y_sample, sim=test_sim, bins=10, version=1)

#         # Resample M* using w_smf
#         resampled_Ms = Corr.weighted_resample(y_sample[:, 0], w_smf)
        
#         # Resample Mh using w_hmf
#         resampled_Mh = Corr.weighted_resample(y_sample[:, 1], w_hmf)
        
#         # Append the resampled M* and Mh values to the lists
#         y_nde_resampled_Ms.append(resampled_Ms)
#         y_nde_resampled_Mh.append(resampled_Mh)

#     # Convert the lists to numpy arrays and combine them into a final array
#     y_nde_resampled_Ms = np.array(y_nde_resampled_Ms)  # Shape: (100, 1000)
#     y_nde_resampled_Mh = np.array(y_nde_resampled_Mh)  # Shape: (100, 1000)

#     # Stack the resampled M* and Mh values to get the final resampled array
#     y_nde_resample = np.stack([y_nde_resampled_Ms, y_nde_resampled_Mh], axis=-1)  # Shape: (100, 1000, 2)

#     return y_nde_resample
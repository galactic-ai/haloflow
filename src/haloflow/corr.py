'''

module to correct for implicity prior imposed on M* 
and Mh from the SMF and HMF. 

'''
import numpy as np 
import warnings 
from . import data as D 


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

    # check ranges of sampled M* and Mh 
    if (Y_sam[:,0].min() < Y_all[:,0].min()) or (Y_sam[:,0].max() > Y_all[:,0].max()):
        warnings.warn('some M* samples are below or above the M* limit')
    if (Y_sam[:,1].min() < Y_all[:,1].min()) or (Y_sam[:,1].max() > Y_all[:,1].max()):
        warnings.warn('some M* samples are below or above the Mh limit')

    # implicit p(M*) prior
    p_smf, ms_bin = np.histogram(Y_all[:,0], bins=bins) 

    # implicit p(Mh) prior
    p_hmf, mh_bin = np.histogram(Y_all[:,1], bins=bins) 
    
    # weights for M* prior 
    ms_sam = (Y_sam[:,0].copy()).clip(ms_bin[0], ms_bin[-1]) # clip the edges 
    w_smf = 1./np.interp(ms_sam, 0.5 * (ms_bin[:-1] + ms_bin[1:]), p_smf) 
    w_smf *= float(len(w_smf)) / np.sum(w_smf) # renormalize (this is for convenience and should not affect anything) 

    # weights for Mh prior 
    mh_sam = (Y_sam[:,1].copy()).clip(mh_bin[0], mh_bin[-1]) # clip the edges 
    w_hmf = 1./np.interp(mh_sam, 0.5 * (mh_bin[:-1] + mh_bin[1:]), p_hmf) 
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
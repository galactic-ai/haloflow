'''

module to interface with simulation data


'''
import os 
import numpy as np 
from astropy.table import Table 


if os.environ['machine'] == 'della': 
    dat_dir = '/scratch/gpfs/chhahn/haloflow/'
else: 
    raise ValueError


def hf2_centrals(dataset, obs, sim='TNG100', version=1): 
    ''' Read Y and X data from subhalos of the haloflow2 simulations.

    
    args: 
        dataset (str): specify training or test data. specify 'train' or 'test'

        obs (str): specify the observables to include 

    '''
    fdata = os.path.join(dat_dir, 'hf2', 'hf2.%s.morph_subhalo.csv' % sim)
    subhalo = Table.read(fdata) 
    
    props = [] 
    if 'mags' in obs: props.append('Sersic_mag') 
    if 'morph' in obs: props += ['Sersic_re', 'Sersic_nser', 'Sersic_axrat']
    if 'morph_extra' in obs: props += ['AsymmetryNoAperture', 'ResidualAsymmetryNoAperture', 'Concentration_Elliptical', 'Smoothness', 'Gini', 'M20', 'SB1kpc']

    cols = []
    for b in ['g', 'r', 'i', 'y', 'z']: 
        for p in props: 
            cols.append('%s_%s' % (p, b))

    if 'satlum' in obs: 
        raise NotImplementedError 
    if 'rich' in obs: 
        raise NotImplementedError

    Y = np.array([np.array(subhalo[col].data) for col in ['log_subhalomass_stars', 'log_subhalomass_dm']]).T # stellar and halo mass 
    X = np.array([np.array(subhalo[col].data) for col in cols]).T
    
    np.random.seed(42) # random seed to the splits are fixed. 
    isort = np.arange(X.shape[0])  
    np.random.shuffle(isort)

    Ntrain = int(0.9 * X.shape[0])
    if dataset == 'train': 
        return Y[isort][:Ntrain], X[isort][:Ntrain]
    elif dataset == 'test': 
        return Y[isort][Ntrain:], X[isort][Ntrain:]
    elif dataset == 'all': 
        return Y[isort], X[isort]


def get_subhalos(dataset, obs, snapshot=91, version=1): 
    ''' see nb/compile_subhalos.ipynb and nb/datasets.ipynb
    '''
    if snapshot != 91: raise NotImpelmentedError  
    fdata  = os.path.join(dat_dir, 'subhalos.central.snapshot%i.v%i.%s.hdf5' % (snapshot, version, dataset))
    
    if os.path.isfile(fdata): 
        subhalo   = Table.read(fdata)
    else: 
        subhalo = Table.read(os.path.join(dat_dir, 'subhalos_morph.hdf5'))
        subhalo = subhalo[subhalo['snapshot'] == snapshot]
        print('%i subhalos' % len(subhalo))
    
        central_id = np.load(os.path.join(dat_dir, 'centrals.subfind_id.snapshot%i.npy' % snapshot))

        is_central = np.array([_id in central_id for _id in subhalo['subhalo_id']])
        subhalo = subhalo[is_central]
        print('%.2f of subhalos are centrals' % np.mean(is_central))
        print('%i subhalos' % len(subhalo))

        
        uid = np.random.choice(np.unique(subhalo['subhalo_id'][subhalo['SubhaloMassType_stars'] > 9.5]), replace=False, size=125)

        i_test = np.zeros(len(subhalo)).astype(bool)
        for _uid in uid:
            i_test[subhalo['subhalo_id'] == _uid] = True
        
        subhalo_test = subhalo[i_test]
        subhalo_train = subhalo[~i_test]

        ftrain  = os.path.join(dat_dir, 'subhalos.central.snapshot%i.train.hdf5' % snapshot)
        ftest  = os.path.join(dat_dir, 'subhalos.central.snapshot%i.test.hdf5' % snapshot)
        subhalo_test.write(ftest) 
        subhalo_train.write(ftrain)
        
        if dataset == 'train': 
            subhalo = subhalo_train 
        elif dataset == 'test': 
            subhalo = subhalo_test
    
    for b in ['g', 'r', 'i', 'z']: 
        subhalo['%s_satlum_all_boxcox' % b] = (subhalo['%s_lum_has_stars' % b]**0.1 - 1)/0.1
        subhalo['%s_satlum_1e9_boxcox' % b] = (subhalo['%s_lum_above_mlim' % b]**0.1 - 1)/0.1
        subhalo['%s_satlum_mr_boxcox' % b]  = (subhalo['%s_lum_above_mrlim' % b]**0.1 - 1)/0.1
    
    props = [] 
    if 'mags' in obs: props.append('Sersic_mag') 
    if 'morph' in obs: props += ['Sersic_Reff', 'CAS_C', 'CAS_A']

    cols = []
    for b in ['g', 'r', 'i', 'y', 'z']: 
        for p in props: 
            cols.append('%s_%s' % (b, p))

    if 'satlum' in obs: 
        for b in ['g', 'r', 'i', 'z']: 
            if 'satlum_all' in obs: 
                cols.append('%s_satlum_all_boxcox' % b)
            elif 'satlum_1e9' in obs: 
                cols.append('%s_satlum_1e9_boxcox' % b)
            elif 'satlum_mr' in obs: 
                cols.append('%s_satlum_mr_boxcox' % b)

    if 'rich' in obs: 
        if 'rich_all' in obs: 
            cols.append('richness_all') 
        elif 'rich_1e9' in obs: 
            cols.append('richness_mlim') 
        elif 'rich_mr' in obs: 
            cols.append('richness_mrlim') 

    y_train = np.array([np.array(subhalo[col].data) for col in ['SubhaloMassType_stars', 'SubhaloMassType_dm']]).T # stellar and halo mass 
    x_train = np.array([np.array(subhalo[col].data) for col in cols]).T

    return y_train, x_train 

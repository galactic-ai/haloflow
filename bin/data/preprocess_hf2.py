#!bin/python 
# python script to preprocess the HF2 data 
import os, sys 
import numpy as np
from astropy.table import Table, vstack
from astropy.table import join as aJoin

from tqdm import trange

sim = sys.argv[1]

if sim not in ['TNG100', 'TNG50', 'Eagle100', 'Simba100', 'TNG50_corr']: 
    raise ValueError('Sim is not included in HF2') 

# for TNG50 you can apply stellar mass and magnitude correction 
corr = False
if 'corr' in sim: 
    sim = 'TNG50'
    corr = True

dat_dir = '/scratch/gpfs/chhahn/haloflow/hf2/'

# read photometry and morphology 
morph = Table.read(os.path.join(dat_dir, 'HaloFlow_%s-1.csv' % sim))
# only keep centrals
morph = morph[morph['GroupFirstSub'] == 1] 
# if flag is 0, the Sersic fit worked and non-parametric morphologies were calculated
morph = morph[morph['ProcessFlag'] == 0]  
print('%i subhalos' % len(morph))

# apply stellar mass, magnitude, surface brightness corrections
if corr: 
    morph['SubhaloMassType_stars']  = morph['SubhaloMassType_stars'] + morph['StellarMassCorrection'] 
    morph['Sersic_mag']             = morph['Sersic_mag'] + morph['MagnitudeCorrection']
    morph['SB1kpc']                 = morph['SB1kpc'] + morph['MagnitudeCorrection']

select_cols = ['SnapNum', 'SubfindID', 'Camera', 'Band',
    'Redshift',
    'SubhaloMassType_stars',    # log10(stellar mass) 
    'Group_M_Crit200',          # log10(halo mass) 
    'Sersic_mag',    # magnitude
    'Sersic_re',     # R_eff
    'Sersic_nser',   # sersic index
    'Sersic_axrat',  # projected minor/major axis ratio (b/a)
    'Sersic_ang',    # position angle
    'Asymmetry',     # asymmetry measured within 1 Petrosian Radius
    'AsymmetryNoAperture', # asymmetry measured using pixels in segmentation footprint
    'RMSAsymmetrySquared', # We could also try RMSAsymmetrySquared in place of the other AsymmetryNoAperture.
    'ResidualAsymmetryNoAperture', # residual asymmetry model-subtractd image of measured using pixels in segmentation footprint
    'Concentration_Elliptical',
    'Smoothness',
    'Gini',
    'M20',
    'SB1kpc']
# only keep useful columns 
for col in morph.colnames:
    if col not in select_cols:
        morph.remove_column(col)

# compile the different bands 
morph_g = morph[morph['Band'] == 'g']
morph_r = morph[morph['Band'] == 'r']
morph_i = morph[morph['Band'] == 'i']
morph_y = morph[morph['Band'] == 'y']
morph_z = morph[morph['Band'] == 'z']

for band, _morph in zip(['g', 'r', 'i', 'y', 'z'], [morph_g, morph_r, morph_i, morph_y, morph_z]):
    for col in _morph.colnames:
        if col not in ['SnapNum', 'SubfindID', 'Camera', 'Band', 'Redshift', 'SubhaloMassType_stars', 'Group_M_Crit200']:
            _morph.rename_column(col, col+'_'+band)
    _morph.remove_column('Band')

for i, _morph in enumerate([morph_g, morph_i, morph_r, morph_y, morph_z]): 
    mask = np.zeros(len(_morph)).astype(bool)
    for k in _morph.columns: 
        try: 
            mask = mask | _morph[k].mask
        except AttributeError: 
            pass
        
    if i == 0: 
        morphs = _morph.copy()
    else:        
        morphs = aJoin(morphs, _morph[~mask], keys=['SnapNum', 'SubfindID', 'Camera', 'Redshift', 'SubhaloMassType_stars', 'Group_M_Crit200'], join_type='left')

# remove any masked data 
mask = np.zeros(len(morphs)).astype(bool)
for col in morphs.colnames:
    flag = (morphs[col] == -99)
    mask = mask | flag
morphs = morphs[~mask]

# write to file 
morphs.write(os.path.join(dat_dir, 'hf2.%s%s.morph_subhalo.csv' % (sim, ['', '_corr'][corr])), overwrite=True)

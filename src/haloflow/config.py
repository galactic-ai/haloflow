import os
import matplotlib as mpl

def get_dat_dir(): 
    ''' get dat_dir based on machine 
    '''
    try:
        if os.environ['machine'] == 'della': 
            return '/scratch/gpfs/chhahn/haloflow/'
        elif os.environ['machine'] == 'puma': 
            return '/xdisk/chhahn/chhahn/haloflow/'
    except KeyError:
        return '../../data/'

def setup_plotting_config():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['axes.xmargin'] = 1
    mpl.rcParams['xtick.labelsize'] = 'x-large'
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.labelsize'] = 'x-large'
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['legend.frameon'] = False

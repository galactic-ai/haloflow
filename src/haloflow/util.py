'''


convenient functions 


'''
import os 
import glob
import numpy as np 

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_best_ndes(study_name, n_ensemble=5, device='cpu', dat_dir='/scratch/gpfs/chhahn/haloflow/nde', verbose=False): 
    ''' read best NDEs from dat_dir
    '''
    fevents = glob.glob(os.path.join(dat_dir, '%s/*/events*' % study_name))
    if verbose: print('%i models trained' % len(fevents))

    events, best_valid = [], []
    for fevent in fevents: 
        ea = EventAccumulator(fevent)
        ea.Reload()

        try: 
            best_valid.append(ea.Scalars('best_validation_log_prob')[0].value)
            events.append(fevent)
        except: 
            pass #print(fevent)

    best_valid = np.array(best_valid)
    
    i_models = [int(os.path.dirname(events[i]).split('.')[-1]) for i 
            in np.argsort(best_valid)[-n_ensemble:][::-1]]
    print(i_models) 
    
    qphis = []
    for i_model in i_models: 
        fqphi = os.path.join(dat_dir, '%s/%s.%i.pt' % (study_name, study_name, i_model))
        qphi = torch.load(fqphi, map_location=device, weights_only=False)
        qphis.append(qphi)

    return qphis

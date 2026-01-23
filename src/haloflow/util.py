'''


convenient functions 


'''
import os 
import glob
import numpy as np 

import torch
import torch.nn as nn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tqdm.auto import tqdm
from joblib import Parallel, delayed
import contextlib
import joblib

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

def get_all_data_from_loader(dataloader):
    all_X, all_Y = [], []
    for batch in dataloader:
        X_batch, Y_batch, _ = batch
        all_X.append(X_batch)
        all_Y.append(Y_batch)

    all_X = torch.cat(all_X, dim=0)
    all_Y = torch.cat(all_Y, dim=0)
    return all_X, all_Y

def weighted_huber_loss(y_true, y_pred, delta=1.0):
    criterion = nn.HuberLoss(delta=delta, reduction='mean')
    loss = criterion(y_pred, y_true)
    weights = 1.0 + (y_true - y_true.min()) / (y_true.max() - y_true.min())  
    return (loss * weights).mean()

def weighted_mse_loss(y_true, y_pred, weights):
    squared_diff = (y_pred - y_true)**2
    loss = torch.mean(squared_diff * weights)
    return loss

def bias_estimate(true, pred, sigma):
    return np.abs(pred - true) / sigma

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()
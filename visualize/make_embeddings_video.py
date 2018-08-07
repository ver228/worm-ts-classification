#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 21:02:42 2018

@author: avelinojaver
"""

import sys
sys.path.append('../../worm-ts-classification')
from worm_ts_classification.flow import SkelTrainer, collate_fn
from worm_ts_classification.path import get_path as get_flow_path

sys.path.append('../../worm-autoencoder/')
from worm_autoencoder.models import AE2D


import torch
from torch.utils.data import DataLoader

if __name__ == '__main__':
    
    fname, _ = get_flow_path('AE_emb32_20180613_l1', platform = 'osx')
    gen = SkelTrainer(fname = fname, 
                              is_balance_training = False,
                              is_tiny = False,
                              is_divergent_set = True,
                              return_label = True, 
                              return_snp = False
                          )
    loader = DataLoader(gen, 
                            batch_size = 1, 
                            collate_fn = collate_fn,
                            num_workers = 1)
    
    
    model_path = '/Volumes/rescomp1/data/WormData/experiments/autoencoders/embeddings/CeNDR_ROIs_embeddings/20180613_174048_AE2D32_l1_adam_lr0.001_batch16/checkpoint.pth.tar' 
    model = AE2D(32)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    
    #%%    
    gen.test()
    for dat in loader:
        break
    #%%
    with torch.no_grad():
        X, _ = dat
        
        res = model.decoder(X[0,0].transpose_(0, 1)[:250])
    #%%
     
#    
#    for v, p in zip(dat.squeeze(), pred.squeeze()):
#        fig, axs = plt.subplots(1,3, figsize=(15, 5), sharex=True, sharey=True)
#        axs[0].imshow(v)
#        axs[1].imshow(p)
#        axs[2].imshow(np.abs(p-v))
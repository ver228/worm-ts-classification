#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 21:02:42 2018

@author: avelinojaver
"""

import sys
sys.path.append('../..')
import flow

sys.path.append('../../worm-autoencoder/src')
import models

import torch
from flow import get_flow_from_sampled
from models import AE2D

if __name__ == '__main__':
    #model_name = '20180613_174048_AE2D32_mse_adam_lr0.001_batch16'
    #model_name = '20180613_174049_AE2D32_l1smooth_adam_lr0.001_batch16'
    #model_name = '20180613_174048_AE2D32_l1_adam_lr0.001_batch16'
    model_name = '20180618_152855_AE2D32_l1_adam_lr0.001_batch16'
    #model_name = '20180618_160914_AE2D32_l1smooth_adam_lr0.001_batch16'
    model_path = '/Volumes/rescomp1/data/WormData/experiments/autoencoders/results/{}/checkpoint.pth.tar'.format(model_name)
    
    model = AE2D(32)
    #loader = get_flow(batch_size=16, epoch_size=10, num_workers=1, sets2include=[2,3,4])
    #loader = get_flow(batch_size=16, epoch_size=10, num_workers=1, sets2include=[2,3,4])
    
    
    loader = get_flow_from_sampled(batch_size = 16, 
                                   epoch_size = None
                                   )
    
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    
    #%%    
    loader.sampler.data_source.test()
    for dat in loader:
        break
    
    with torch.no_grad():
        pred = model(dat)
    
    for v, p in zip(dat.squeeze(), pred.squeeze()):
        fig, axs = plt.subplots(1,3, figsize=(15, 5), sharex=True, sharey=True)
        axs[0].imshow(v)
        axs[1].imshow(p)
        axs[2].imshow(np.abs(p-v))
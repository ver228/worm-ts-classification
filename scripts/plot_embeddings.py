#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:19:19 2018

@author: avelinojaver
"""

import sys
sys.path.append('../../worm-ts-classification')

sys.path.append('../../worm-autoencoder/')
from worm_autoencoder.models import AE2D, AE2DWithSkels
from worm_autoencoder.flow import FlowFromSampledSkels

import torch
import tables
from pathlib import Path
import torch
#%%
if __name__ == '__main__':
    
    
    
    #%%
    emb_model_root = Path('/Users/avelinojaver/Data/experiments/autoencoders/results')
    AE_models_data = [('AE_emb32_20180613_l1' ,  AE2D, '20180613_174048_AE2D32_l1_adam_lr0.001_batch16'),
                ('AE2DWithSkels32_emb32_20180620', AE2DWithSkels,'20180620_173601_AE2DWithSkels32_skel-1-1_adam_lr0.001_batch16'),
                ]
    
    AE_models = {} 
    for emb_t, mod_func, mod_name in AE_models_data:
        model_path = emb_model_root / mod_name /  'checkpoint.pth.tar'
    
    
        model = mod_func(32)
        state = torch.load(model_path, map_location = 'cpu')
        model.load_state_dict(state['state_dict'])
        
        AE_models[emb_t] = model
    
    fname = '/Users/avelinojaver/Data/experiments/autoencoders/data/sampled/CeNDR_20180615_164245.hdf5'
    gen = FlowFromSampledSkels(main_file=fname, is_shuffled = False)
    gen.test()
    #%%
    tot_imgs = 15
    
    step = len(gen)//(tot_imgs-1)
    
    ll = [27558, 60001, 18878, 30295, 684,  95411]
    #ll = [random.randint(0, len(gen)) for _ in range(tot_imgs)]
    
    print(ll)
    
    X = []
    skels = []
    for ii in ll:
        x, skel = gen[ii]
        X.append(x)
        skels.append(skel)
        
        #plt.figure()
        #plt.imshow(x.squeeze(), vmin=bot, vmax=top, cmap='gray')
        #ss = (skel+0.5)*128
        #plt.plot(ss[:,0], ss[:,1], 'r')
    X = np.stack(X)
    
    Xc = torch.from_numpy(X)
    Xhats = {}
    with torch.no_grad():
        for emb_t, model in AE_models.items():
            emb = model.encoder(Xc)
            Xhat = model.decoder(emb)
            Xhats[emb_t] = Xhat.numpy()[:, 0]
     
    X0 = X[:, 0]
    X1 = Xhats['AE_emb32_20180613_l1']
    #X2 = Xhats['AE2DWithSkels32_emb32_20180620']
    
    figsize = (X.shape[0]*2.8, 6)
    
    bot = np.min(X0)
    top = np.max(X0)
    fig, axs = plt.subplots(2, X.shape[0], figsize = figsize, sharex=True, sharey=True)
    for ii in range(X0.shape[0]):
        x0 = X0[ii]
        x1 = X1[ii]
        
        axs[0][ii].imshow(x0, vmin=bot, vmax=top, cmap='gray')
        axs[0][ii].axis('off')
        
        axs[1][ii].imshow(x1, vmin=bot, vmax=top, cmap='gray')
        axs[1][ii].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig('AE_results.pdf')
    
    fig, axs = plt.subplots(2, X.shape[0], figsize = figsize, sharex=True, sharey=True)
    for ii, (skel, x0, x1) in enumerate(zip(skels, X0, X1)):
        ss = (skel+0.5)*128
        cc = np.round(ss[5]).astype(np.int)
        
        off = 12
        
        x0s = x0[cc[1]-off:cc[1]+off, cc[0]-off:cc[0]+off]
        axs[0][ii].imshow(x0s, vmin=bot, vmax=top, cmap='gray')
        axs[0][ii].axis('off')
        
        x1s = x1[cc[1]-off:cc[1]+off, cc[0]-off:cc[0]+off]
        axs[1][ii].imshow(x1s, vmin=bot, vmax=top, cmap='gray')
        axs[1][ii].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig('AE_zoomed.pdf')
    
    #%%
    if False:
        emb_types = [
            'skeletons',
            'AE_emb32_20180613_l1', 
            'AE2DWithSkels32_emb32_20180620'
            ]
    
        row_id = 25
        max_frame = 2500
        data_x = {}
        generators = {}
        for emb_t in emb_types:
            #fname, results_dir_root = get_path(emb_t)
            fname = '/Users/avelinojaver/Data/experiments/classify_strains/CeNDR_{}.hdf5'.format(emb_t)
            gen = SkelTrainer(fname = fname,
                              is_divergent_set = False, 
                              is_tiny = False,
                                return_label = False, 
                                return_snp = False,
                                unsampled_test = True,
                                sample_size = 22500,
                                )
            gen.test()
            X, = gen[row_id]
            
            data_x[emb_t] = X
            generators[emb_t] = gen
        #%%
        max_frame = 22500
        
        Xhats = {}
        for emb_t, model in AE_models.items():
        
            X = data_x[emb_t]
            
            with torch.no_grad():
                xx = X[0, :, ::100].T
                xx = torch.from_numpy(xx)
                
                Xhat = model.decoder(xx)
                Xhats[emb_t] = Xhat.numpy()[:, 0]
                if 'skel' in emb_t.lower():   
                    skelshat = model.regression(xx)
        #%%
        X1 = Xhats['AE_emb32_20180613_l1']
        X2 = Xhats['AE2DWithSkels32_emb32_20180620']
        for ii in range(0, X1.shape[0], 100):
            fig, axs = plt.subplots(1,2, figsize=(12, 12) , sharex=True, sharey=True)
            axs[0].imshow(X1[ii])
            axs[1].imshow(X2[ii])
        
        
    
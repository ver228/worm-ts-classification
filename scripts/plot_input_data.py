#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:19:19 2018

@author: avelinojaver
"""
import sys
sys.path.append('../../worm-ts-classification')

from worm_ts_classification.flow import SkelTrainer, collate_fn, DIVERGENT_SET
from worm_ts_classification.path import get_path

from torch.utils.data import DataLoader
import tqdm
#%%
if __name__ == '__main__':
    emb_types = [
            'angles',
            'skeletons',
            'eigen',
            'eigenfull',
            'AE_emb32_20180613_l1', 
            'AE2DWithSkels32_emb32_20180620'
            ]
    #%%
    generators = {}
    for emb_t in emb_types:
        fname, results_dir_root = get_path(emb_t)
        #fname = '/Users/avelinojaver/Data/experiments/classify_strains/CeNDR_{}.hdf5'.format(emb_t)
        generators[emb_t] = SkelTrainer(fname = fname,
                          is_divergent_set = False, 
                          is_tiny = False,
                            return_label = False, 
                            return_snp = False,
                            unsampled_test = True,
                            sample_size = 22500,
                            )
        
    
    
    plot_params = {
            'angles' :('Angles', list(range(0, 49, 8)), None),
            'eigen' : ('EigenWorms', list(range(0, 6)), list(range(1, 7))), 
            'AE_emb32_20180613_l1' : ('AutoEncoder', list(range(0, 33, 8)), None),
            #'AE2DWithSkels32_emb32_20180620' : ('AutoEncoder + Skels', list(range(0, 33, 8)), None)
            }
    #%%
    row_id = 25
    max_frame = 2500
    tot_subplots = len(plot_params) + 1
    fig, axs = plt.subplots(tot_subplots, 1, figsize=(3*tot_subplots, 6), sharex=True)
    
    
    fig.text(0.07, 0.42, 'Number of Embeddings', va='center', rotation='vertical',  fontsize=16)
    fig.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=16)
    
    
    xx = 0.135
    y_ini = 0.26
    y_fin= 0.85
    y_shift = (y_fin-y_ini)/(tot_subplots-1)
    for ii in range(tot_subplots):
        tt = chr(ord('A') + ii)
        yy = y_fin-y_shift*ii
        
        cc =  'black' if ii == 0 else 'white'
        fig.text(xx, yy,  tt, color=cc, ha='center', fontsize=16)
    
    
    for ii, emb_t in enumerate(['angles', 'eigen', 'AE_emb32_20180613_l1']):#, 'AE2DWithSkels32_emb32_20180620']):
        
        gen = generators[emb_t]
        gen.test()
        
        stitle, syticks, sytickslabels = plot_params[emb_t]
        if sytickslabels is None:
            sytickslabels = syticks.copy()    
            sytickslabels[0] = 1
        
        X, = gen[row_id]
        X = X[0, :, :max_frame]
        
        axs[ii+1].imshow(X, aspect='auto', interpolation='none')
        #axs[ii+1].set_title(stitle)
        
        axs[ii+1].set_yticks(syticks)
        axs[ii+1].set_yticklabels(sytickslabels, fontsize=10)
        
    
    
    gen = generators['skeletons']
    gen.test()
    X, = gen[row_id]
    X = X[0, :, :max_frame]
    
    skel_x = X[:49]
    skel_y = X[98:48:-1]
    
    offset = 1
    magnifier = 125
    
    sylim = (-120, 120)
    for ii in list(range(magnifier//2, max_frame, magnifier)):
        xx = skel_x[:, ii]*magnifier + offset*ii 
        yy = skel_y[:, ii]*magnifier
        axs[0].plot(xx, yy, color='navy', lw=2.5)
        axs[0].plot(xx[0], yy[0], 'o', ms=3, color='tomato')
        axs[0].plot((ii,ii), sylim, ':k')
        
    axs[0].set_ylim(*sylim)
    axs[0].set_yticks([])
    axs[0].axis('off')
    
    axs[-1].set_xticks(list(range(0, max_frame+1, 500)))
    axs[-1].set_xticklabels(np.arange(0, (max_frame/25)+1, 500/25), fontsize=12)
    axs[-1].set_xlim(0, max_frame)
    
    fig.savefig('embeddings.pdf')
    
    #%%
    
    max_frame = 1000
    emb_t = 'AE_emb32_20180613_l1'
    gen = generators[emb_t]
    gen.test()
    
    stitle, syticks, sytickslabels = plot_params[emb_t]
    if sytickslabels is None:
        sytickslabels = syticks.copy()    
        sytickslabels[0] = 1
    
    X, = gen[row_id]
    X = X[0, :, :max_frame]
    
    
    emb2plot = [0, 15, 31]
    #emb2plot = list(range(3, 32, 4))
    
    tot_subplots = len(emb2plot)
    fig, axs = plt.subplots(tot_subplots, 1, figsize=(5*tot_subplots, 3), sharex=True, sharey=True)
    xx = 0.13
    y_ini = 0.13
    y_fin= 0.67
    y_shift = (y_fin-y_ini)/(tot_subplots-1)
    for ii, n_emb in enumerate(emb2plot):
        axs[ii].plot(X[n_emb, :])
        
        tt = 'Emb={}'.format(n_emb+1)
        yy = y_fin-y_shift*ii
        fig.text(xx, yy,  tt, color='black', ha='left', fontsize=11)
        
    fig.text(0.5, 0.02, 'Time (s)', ha='center', fontsize=13)
    fig.text(0.07, 0.5, 'Value', va='center', rotation='vertical',  fontsize=13)
    
    
    plt.ylim(-1.2, 1.2)
    fig.savefig('AE_embeddings_ts.pdf')
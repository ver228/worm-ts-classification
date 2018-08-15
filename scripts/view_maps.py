#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:31:12 2018

@author: avelinojaver
"""
import sys
sys.path.append('../../worm-ts-classification')
from worm_ts_classification.flow import DIVERGENT_SET, load_strain_dict
from worm_ts_classification.path import get_path

import pandas as pd
import numpy as np
import os
import pickle
import tqdm

from pathlib import Path


if __name__ == '__main__':
    
    
    set_type = 'angles'
    #set_type = 'AE_emb_20180206'
    
    _, results_dir_root = get_path(set_type)
    #model_path = 'logs/angles_20180531_125503_R_simpledilated_sgd_lr0.0001_wd0_batch8'
    #model_path = 'log_divergent_set/AE2DWithSkels32_emb32_20180620_20180622_121637_simpledilated1d_div_adam_lr1e-05_wd0_batch8'
    #model_path = 'log_divergent_set/AE_emb32_20180613_l1_20180620_002605_simpledilated1d_div_adam_lr0.0001_wd0_batch8'
    model_path = 'logs/SWDB_angles_20180627_184430_simpledilated_sgd_lr0.001_wd0.0001_batch8'
    
    model_path = os.path.join(results_dir_root, model_path, 'model_best.pth.tar')
    
    
    bn = model_path.split(os.sep)[-2]
    save_path = os.path.join(results_dir_root, 'maps_clf', bn)
    
    save_path = Path(save_path)
    
    save_name = os.path.join(save_path, 'clf_results.csv')
    df = pd.read_csv(save_name)
    
    #%%
    #strain_dict = load_strain_dict()
    #divergent_set_ids = [strain_dict[x] for x in DIVERGENT_SET]
    #good = df['target'].isin(divergent_set_ids)
    
    good = df['target_strain'] == 'N2_XX'
    
    df_div = df[good]
    acc_div = np.mean(df_div['target'] == df_div['top1'])
    print(acc_div)
    print(df_div['isin_top5'].mean())
    
    df_res= df[~good]
    acc_res = np.mean(df_res['target'] == df_res['top1'])
    print(acc_res)
    print(df_res['isin_top5'].mean())
    #%%
    import seaborn as sns
    cc = pd.crosstab(df_div['target'], df_div['top1'], normalize='columns')
    sns.heatmap(cc, cmap='plasma')
    
    dd = {v:k for k,v in strain_dict.items()}
    cc.columns = [dd[x] for x in cc.columns] 
    cc.index = [dd[x] for x in cc.index] 
#%%    
#    #%%
#    all_res = []
#    
#    fnames = list(save_path.rglob('maps_vid*.p'))
#    for fname in tqdm.tqdm(fnames):
#        with open(fname, 'rb') as fid:
#            data = pickle.load(fid)
#        #%%
#        maps = data['maps']
#        target = data['target']
#        m_act = np.mean(maps, axis = tuple(range(1,maps.ndim)))
#        m_act_e = np.exp(m_act)
#        pred = m_act_e/np.sum(m_act_e)
#        
#        
#        top5 = np.argsort(pred)[-5:][::-1]
#        
#        top1 = top5[0]
#        pred_v = pred[top1]
#        
#        isin_top5 = data['target'] in top5
#        
#        
#        dd = (target, top1, pred_v, isin_top5)
#        all_res.append(dd)
#        #%%
#        maps_e = np.exp(maps)
#        maps_S = maps_e/np.sum(maps_e)
#        #%%
#        #fname_X = fname.parent / fname.name.replace('maps_', 'X_')
#        #with open(fname_X, 'rb') as fid:
#        #    X = pickle.load(fid)
        
    #%%
    #df = pd.DataFrame(all_res, columns=['target', 'top1', 'confidence', 'isin_top5'])
    #%%
#    strain_dict = load_strain_dict()
#    divergent_set_ids = [strain_dict[x] for x in DIVERGENT_SET]
#    good = df['target'].isin(divergent_set_ids)
#    
#    df_div = df[good]
#    acc_div = np.mean(df_div['target'] == df_div['top1'])
#    print(acc_div)
#    print(df_div['isin_top5'].mean())
#    
#    df_res= df[~good]
#    acc_res = np.mean(df_res['target'] == df_res['top1'])
#    print(acc_res)
#    print(df_res['isin_top5'].mean())
#    #%%
#    import seaborn as sns
#    cc = pd.crosstab(df_div['target'], df_div['top1'], normalize='columns')
#    sns.heatmap(cc, cmap='plasma')
#    
#    dd = {v:k for k,v in strain_dict.items()}
#    cc.columns = [dd[x] for x in cc.columns] 
#    cc.index = [dd[x] for x in cc.index] 
    #%%
#    fname = os.path.join(save_path, 'maps.p')
#    
#    
#    
#    
#        
#    np.mean(data['targets'] == data['predictions'])
#    #%%
#    map_v = data['maps'][0].squeeze()
##%%
#    avgs = []
#    for map_v in data['maps']:
#        map_v = map_v.squeeze()
#        
#        seg_mean  = np.mean(map_v, axis=1)
#        
#        ii = np.argmax(seg_mean)
#        dd = ii, seg_mean[ii], seg_mean[0]/seg_mean[ii], seg_mean[-1]/seg_mean[ii]
#        avgs.append(dd)
#        plt.plot(seg_mean)
#    #%%
#    avgs = np.array(avgs)
    
        #%%
    #     Xf = torch.from_numpy(gen._full_video(ind).T[None, None])
    #     Xf = Xf.to(device)
    #     print('F:', model(Xf).max(1))
        
    #     for _ in range(3):
            
    #         Xs = torch.from_numpy(gen._sample_video(ind).T[None, None])
    #         Xs = Xs.to(device)
    #         print('S:', model(Xs).max(1))

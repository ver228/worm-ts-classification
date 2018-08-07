#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:31:12 2018

@author: avelinojaver
"""
#from pathlib import Path
#import sys
#source_path = Path(__file__).resolve().parent
#sys.path.append(str(source_path))

from path import get_path
from trainer import get_model

import pandas as pd
import os
import pickle
import torch
import numpy  as np

from flow import SkelTrainer, DIVERGENT_SET
import tqdm



if __name__ == '__main__':
    cuda_id = 2
    
    set_type = 'SWDB_angles'
    model_path = 'logs/SWDB_angles_20180627_184430_simpledilated_sgd_lr0.001_wd0.0001_batch8'
    
    fname, results_dir_root = get_path(set_type)

    model_path = os.path.join(results_dir_root, model_path, 'model_best.pth.tar')

    
    bn = model_path.split(os.sep)[-2]
    
    parts = bn[len(set_type) + 1:].split('_')
    
    model_name = parts[3] if parts[2] == 'R' else parts[2]
    
    if torch.cuda.is_available():
        dev_str = "cuda:" + str(cuda_id)
        print("THIS IS CUDA!!!!")
    else:
        dev_str = 'cpu'

    print(dev_str)
    device = torch.device(dev_str)

    return_label = True
    return_snp = False
    unsampled_test = True
    is_divergent_set = 'divergent_set' in model_path

    gen = SkelTrainer(fname = fname,
                      is_divergent_set = is_divergent_set,
                      return_label = return_label,
                      return_snp = return_snp,
                      unsampled_test = unsampled_test)

    model = get_model(model_name, gen.num_classes, gen.embedding_size)
    model = model.to(device)
    
    assert set_type in model_path
    state = torch.load(model_path, map_location = dev_str)
    model.load_state_dict(state['state_dict'])
    model.eval()

    gen.test()
    test_indexes = gen.valid_index
    
    all_res = []
    with torch.no_grad():
        pbar = tqdm.tqdm(test_indexes)
        for vid_id in pbar:
            strain = gen.video_info.loc[vid_id, 'strain']
            target = gen._strain_dict[strain]
            #%%
            x_in = gen._get_data(vid_id)[None, None, :, :]
            X = torch.from_numpy(x_in).to(device)
            
            pred = model(X)
            
            
            
            top5 = np.argsort(pred)[-5:][::-1]
            
            top1 = top5[0]
            pred_v = pred[top1]
            
            isin_top5 = target in top5
            
            
            dd = (target, top1, pred_v, isin_top5)
            all_res.append(dd)
            break
    #%%
    
    df = pd.DataFrame(all_res, columns=['target', 'top1', 'confidence', 'isin_top5'])
    strain_dict_ids = {x:k for k,x in gen._strain_dict.items()}
    
    df['target_strain'] = df['target'].map(strain_dict_ids)
    save_name = os.path.join(save_path, 'clf_results.csv')
    df.to_csv(save_name)
    
    
    divergent_set_ids = [gen._strain_dict[x] for x in DIVERGENT_SET]
    good = df['target'].isin(divergent_set_ids)
    
    df_div = df[good]
    acc_div = np.mean(df_div['target'] == df_div['top1'])
    print(acc_div)
    print(df_div['isin_top5'].mean())
    
    
    
    df_res= df[~good]
    if len(df_res):
        acc_res = np.mean(df_res['target'] == df_res['top1'])
        print(acc_res)
        print(df_res['isin_top5'].mean())
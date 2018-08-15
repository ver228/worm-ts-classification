#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:31:12 2018

@author: avelinojaver
"""
import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from worm_ts_classification.path import get_path
from worm_ts_classification.trainer import get_model
from worm_ts_classification.flow import SkelTrainer

import pandas as pd
import os
import torch

import tqdm

def get_embeddings(save_name, set_type, model_path):   
    fname, results_dir_root = get_path(set_type)
    
    model_path = os.path.join(results_dir_root, model_path, 'checkpoint.pth.tar')

    
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
    print(state['epoch'])
    model.load_state_dict(state['state_dict'])
    model.eval()

    gen.test()
    test_indexes = gen.valid_index
    
    all_embeddings = []
    with torch.no_grad():
        pbar = tqdm.tqdm(test_indexes)
        for vid_id in pbar:
            strain = gen.video_info.loc[vid_id, 'strain']
            
            x_in = gen._get_data(vid_id)[None, None, :, :]
            X = torch.from_numpy(x_in).to(device)
            
            maps = model.cnn_clf(X)
            emb = model.fc_clf[:1](maps).squeeze()
            
            emb = emb.cpu().detach().numpy()
            
            dd = (strain, *emb)
            all_embeddings.append(dd)
    df = pd.DataFrame(all_embeddings)
    
    fname = save_dir / (save_name + '_embeddings.csv')
    df.to_csv(str(fname))
#%%
if __name__ == '__main__':
   cuda_id = 2
   save_dir = ''

   all_args = [
            ('CeNDR_angles',
             'angles',
             'log_divergent_set/angles_20180524_115242_simpledilated_div_lr0.0001_batch8'
             ),
            ('SWDB_angles',
             'SWDB_angles',
             'log_SWDB_angles/SWDB_angles_20180711_214814_R_simpledilated_sgd_lr0.0001_wd0.0001_batch8'
             ),
           ]
   
   
   _, results_dir_root = get_path('')
   save_dir = Path(results_dir_root) / 'summary'
   save_dir.mkdir(parents=True, exist_ok=True)
   
   for args in all_args:
       get_embeddings(*args)
       
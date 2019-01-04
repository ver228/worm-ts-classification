#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:31:12 2018

@author: avelinojaver
"""
import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from worm_ts_classification.path import get_path
from worm_ts_classification.trainer import get_model
from worm_ts_classification.flow import SkelTrainer

import pandas as pd
import os
import torch

import tqdm

#%%
def get_embeddings(save_name, set_type, model_path, argkws = {}):   
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

    
    return_snp = 'snp' in model_path
    return_label = not return_snp
    
    unsampled_test = True
    is_divergent_set = 'divergent_set' in model_path
    
    gen = SkelTrainer(fname = fname,
                      is_divergent_set = is_divergent_set,
                      return_label = return_label,
                      return_snp = return_snp,
                      unsampled_test = unsampled_test,
                      train_epoch_magnifier=1,
                      **argkws)


    
    
    
    #assert set_type in model_path
    state = torch.load(model_path, map_location = dev_str)
    print(state['epoch'])
    
    
    kk = next(reversed(state['state_dict'].keys()))
    model_num_classes = state['state_dict'][kk].shape[0]
    
    
    model = get_model(model_name, model_num_classes, gen.embedding_size)
    model = model.to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    
    if save_name.endswith('_TRAIN'):
        gen.train()
        valid_indexes = gen.valid_index
    elif save_name.endswith('_ALL'):
        
        gen.train()
        v1 = gen.valid_index.copy()
        
        gen.test()
        v2 = gen.valid_index.copy()
        
        valid_indexes = v1 + v2
        
    else:
        gen.test()

    
    
    
    
    all_embeddings = []
    with torch.no_grad():
        pbar = tqdm.tqdm(valid_indexes)
        for vid_id in pbar:
            strain = gen.video_info.loc[vid_id, 'strain']
            
            x_in = gen._get_data(vid_id)[None, None, :, :]
            X = torch.from_numpy(x_in).to(device)
            
            try:
                maps = model.cnn_clf(X)
            except AttributeError:
                maps = model.features(X.repeat(1,3,1,1))
            
            emb = model.fc_clf[:1](maps).squeeze()
            
            emb = emb.cpu().detach().numpy()
            
            dd = (strain, *emb)
            all_embeddings.append(dd)
    df = pd.DataFrame(all_embeddings)
    
    fname = save_dir / (save_name + '_embeddings.csv')
    df.to_csv(str(fname))
#%%
if __name__ == '__main__':
   cuda_id = 1
   save_dir = ''

   all_args = [
#            ('CeNDR_div_angles',
#             'angles',
#             'log_divergent_set/angles_20180524_115242_simpledilated_div_lr0.0001_batch8'
#             ),
#            ('SWDB_angles',
#             'SWDB_angles',
#             'log_SWDB_angles/SWDB_angles_20180711_214814_R_simpledilated_sgd_lr0.0001_wd0.0001_batch8'
#             ),
#             ('CeNDR_angles_TRAIN',
#             'angles',
#             'log_CeNDR/angles_20180531_125503_R_simpledilated_sgd_lr0.0001_wd0_batch8'
#             ),
#            ('SWDB_angles_TRAIN',
#             'SWDB_angles',
#             'log_SWDB_angles/SWDB_angles_20180711_214814_R_simpledilated_sgd_lr0.0001_wd0.0001_batch8'
#             ),
#             ('CeNDR_angles',
#             'angles',
#             'log_CeNDR/angles_20180531_125503_R_simpledilated_sgd_lr0.0001_wd0_batch8'
#             ),
#            ('CeNDR_angles_snp',
#             'angles',
#             'logs_snp/angles_20180524_222950_simpledilated_lr0.0001_batch8'
#             )
#             ('SWDB_angles_resnet18',
#             'SWDB_angles',
#             'logs/SWDB_angles_20180808_143338_resnet18_sgd_lr0.0001_wd0.0001_batch8'
#             ),
#             ('CeNDR_angles_resnet18',
#             'angles',
#             'logs/angles_20180819_084342_R_resnet18_sgd_lr0.0001_wd0.0001_batch8'
#             ),
#             ('SWDB_using_CeNDR',
#              'SWDB_angles',
#              'log_CeNDR/angles_20180531_125503_R_simpledilated_sgd_lr0.0001_wd0_batch8'
#              ),
#              ('CeNDR_using_SWDB',
#              'CeNDR_angles',
#              'log_SWDB_angles/SWDB_angles_20180711_214814_R_simpledilated_sgd_lr0.0001_wd0.0001_batch8'
#              ),
#             ('CeNDRAgg_using_CeNDR_ALL',
#              'CeNDRAgg_angles',
#              'log_CeNDR/angles_20180531_125503_R_simpledilated_sgd_lr0.0001_wd0_batch8'
#              ),
#              ('CeNDRAgg_using_SWDB_ALL',
#              'CeNDRAgg_angles',
#              'log_SWDB_angles/SWDB_angles_20180711_214814_R_simpledilated_sgd_lr0.0001_wd0.0001_batch8'
#              ),
           ]
   
   t_window = 22500
   t_delta = 3750
   max_t = 67500
   #%%
   for t_ini in range(0, max_t-t_window+1, t_delta):
       
       initial_frame, last_frame  = (t_ini, t_ini + t_window)
       dd = (f'CeNDRAgg-t{initial_frame:05d}-{last_frame:05d}_using_CeNDR_ALL',
              'CeNDRAgg_angles',
              'log_CeNDR/angles_20180531_125503_R_simpledilated_sgd_lr0.0001_wd0_batch8',
              {'initial_frame' : initial_frame, 'last_frame' : last_frame}
              )
       all_args.append(dd)
   
   #%%
   _, results_dir_root = get_path('')
   save_dir = Path(results_dir_root) / 'summary'
   save_dir.mkdir(parents=True, exist_ok=True)
   #%%
   for args in all_args:
       get_embeddings(*args)
       
       
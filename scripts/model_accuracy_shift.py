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
from worm_ts_classification.flow import SkelTrainer, DIVERGENT_SET

from sklearn.metrics import f1_score

import datetime
import pandas as pd
import os
import torch
import numpy  as np

import tqdm

def get_accuracies(save_name, set_type, model_path, time_ranges, is_divergent_set, cuda_id=0):   
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
    #is_divergent_set = 'divergent_set' in model_path
    
    
    gen = SkelTrainer(fname = fname,
                      is_divergent_set = is_divergent_set,
                      return_label = return_label,
                      return_snp = return_snp,
                      unsampled_test = unsampled_test,
                      fold_n_test=-1,
                      train_epoch_magnifier = 1,
                      initial_frame = time_ranges[0], 
                      last_frame = time_ranges[1])

    gen.train()

    model = get_model(model_name, gen.num_classes, gen.embedding_size)
    model = model.to(device)
    
    #assert set_type in model_path
    state = torch.load(model_path, map_location = dev_str)
    print(state['epoch'])
    model.load_state_dict(state['state_dict'])
    model.eval()


    
    #model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
    
    
    test_indexes = gen.valid_index
    
    all_res = []
    with torch.no_grad():
        pbar = tqdm.tqdm(test_indexes)
        for vid_id in pbar:
            strain = gen.video_info.loc[vid_id, 'strain']
            target = gen._strain_dict[strain]
            
            x_in = gen._get_data(vid_id)[None, None, :, :]
            
            X = torch.from_numpy(x_in).to(device)
            
            try:
                
                pred = model(X)
            except:
                continue
            pred = pred.cpu().detach().numpy()[0]
            
            
            top5 = np.argsort(pred)[-5:][::-1]
            
            top1 = top5[0]
            pred_v = pred[top1]
            
            isin_top5 = target in top5
            
            
            dd = (target, top1, pred_v, isin_top5)
            all_res.append(dd)
            
    #%%
    
    df = pd.DataFrame(all_res, columns=['target', 'top1', 'confidence', 'isin_top5'])
    strain_dict_ids = {x:k for k,x in gen._strain_dict.items()}
    df['target_strain'] = df['target'].map(strain_dict_ids)
    
    if 'divergent' in model_path:
        divergent_set_ids = [gen._strain_dict[x] for x in DIVERGENT_SET]
        good = df['target'].isin(divergent_set_ids)
        df = df[good]
        
    #save_name = os.path.join(save_path, 'clf_results.csv')
    #df.to_csv(save_name)
        
        
    #%%
    acc_top1 = (df['top1']==df['target']).mean()
    acc_top5 = df['isin_top5'].mean()
    f1 = f1_score(df['target'], df['top1'], average='macro')
    
    summary = (save_name, acc_top1, acc_top5, f1)
    
    return summary, df
#%%
if __name__ == '__main__':
   cuda_id = 2
   save_dir = ''
   
   
   save_name = 'CeNDR_on_CeNDRAgg'
   set_type = 'CeNDRAgg_angles'
   model_path = 'log_CeNDR/angles_20180531_125503_R_simpledilated_sgd_lr0.0001_wd0_batch8'
   

   _, results_dir_root = get_path('')
   save_dir = Path(results_dir_root) / 'summary'
   save_dir.mkdir(parents=True, exist_ok=True)
   
   all_summaries = []
   
   t_window = 22500
   t_delta = 3750
   max_t = 67500
   is_divergent_set = True
   for t_ini in range(0, max_t-t_window+1, t_delta):
       summary, df = get_accuracies(save_name, 
                                    set_type,
                                    model_path,
                                    is_divergent_set = is_divergent_set,
                                    time_ranges = (t_ini, t_ini + t_delta),
                                    cuda_id=cuda_id)
       
       bn = '{}_{}s'.format(save_name, int(round(t_ini/25.)))
       df.to_csv(save_dir / (bn + '.csv'))
       
       print(summary)
       
       all_summaries.append(summary)
   
   all_summaries = pd.DataFrame(all_summaries, columns=['name', 'acc_top1', 'acc_top5', 'f1'])
   
    
   date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   save_name = save_dir / 'Summary_{}.csv'.format(date_str)
   all_summaries.to_csv(str(save_name))

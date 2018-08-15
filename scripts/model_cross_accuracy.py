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
from worm_ts_classification.flow import SkelTrainer, STRAINS_IN_COMMON 

from sklearn.metrics import f1_score

import pandas as pd
import os
import torch
import numpy  as np

import tqdm



#%%
if __name__ == '__main__':
   cuda_id = 2
   save_dir = ''

   set_data = { 
            'SWDB':(
             'SWDB_angles',
             'logs/SWDB_angles_20180807_192205_simpledilated_WTcommon_sgd_lr0.001_wd0.0001_batch8'
             #'logs/SWDB_angles_20180717_111958_R_simpledilated_week_sgd_lr0.001_wd0.0001_batch8'
             #'log_SWDB_angles/SWDB_angles_20180711_214814_R_simpledilated_sgd_lr0.0001_wd0.0001_batch8'
             ),
            
            
            'CeNDR':(
             'angles',
             'logs/angles_20180808_104650_simpledilated_WTcommon_adam_lr0.0001_wd0_batch8'
             #'log_divergent_set/angles_20180524_115242_simpledilated_div_lr0.0001_batch8'
             )
             
            }
   
   
   _, results_dir_root = get_path('')
   save_dir = Path(results_dir_root) / 'summary'
   save_dir.mkdir(parents=True, exist_ok=True)
   
   #%%
   generators = {}
   models = {}
   for save_name, (set_type, model_path) in set_data.items():
       fname, results_dir_root = get_path(set_type)
       gen = SkelTrainer(fname = fname,
                          is_common_WT = True,
                          return_label = True,
                          unsampled_test = True)
       generators[save_name] = gen
      
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
    
       
       model = get_model(model_name, gen.num_classes, gen.embedding_size)
       model = model.to(device)
       
       state = torch.load(model_path, map_location = dev_str)
       model.load_state_dict(state['state_dict'])
       model.eval()
       
       
       strain_dict_r = {v:k for k,v in gen._strain_dict.items()}
       models[save_name] = (model, strain_dict_r)
       
   
   #%%
   for trained_set, test_set in [('CeNDR', 'SWDB'), ('SWDB', 'CeNDR')]:
       gen_test = generators[test_set]
       gen_test.test()
       test_indexes = gen_test.valid_index
    
       all_res = []
       with torch.no_grad():
            pbar = tqdm.tqdm(test_indexes)
            for vid_id in pbar:
                target_strain = gen_test.video_info.loc[vid_id, 'strain']
                #target = gen._strain_dict[strain]
                
                x_in = gen_test._get_data(vid_id)[None, None, :, :]
                X = torch.from_numpy(x_in).to(device)
                
                
                model, strain_dict_r = models[trained_set]
                pred = model(X)
                pred = pred.cpu().detach().numpy()[0]
                predicted_strain_diff = strain_dict_r[np.argmax(pred)]
                
                model, strain_dict_r = models[test_set]
                pred = model(X)
                pred = pred.cpu().detach().numpy()[0]
                predicted_strain_same = strain_dict_r[np.argmax(pred)]
                
                
                dd = (target_strain, predicted_strain_same, predicted_strain_diff)
                all_res.append(dd)
                
            
       #%%
       df = pd.DataFrame(all_res, columns=['target_strain', 'predicted_strain_same', 'predicted_strain_diff'])
       save_name = save_dir / 'S_Base={}-Diff={}.csv'.format(trained_set, test_set)
       df.to_csv(str(save_name))
#%%
#       summary, df = get_accuracies(*args)
#       
#       df.to_csv(save_dir / (args[0] + '.csv'))
#       
#       print(summary)
#       
#       all_summaries.append(summary)
#       
#   #all_summaries = pd.DataFrame(all_summaries, columns=['name', 'acc_top1', 'acc_top5', 'f1'])
#   #all_summaries.to_csv(df.to_csv(save_dir / (args[0] + '.csv')))
#   #%%
#   def func(x):
#      if isinstance(x, str):
#          return x
#      else:
#          return "{:10.4f}".format(x)
#   
#   for x in dd:
#       x = list(x)
#       x[1] = x[1] * 100
#       x[2] = x[2] * 100
#       ss = '{} & {:10.2f} & {:10.2f} & {:10.4f}'.format(*x)
#       
#       print(ss + r'\\')
#   
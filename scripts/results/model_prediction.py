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
print(root_dir)

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

#'checkpoint.pth.tar'
def get_predictions(save_name, 
                    set_type, 
                    model_path, 
                    cuda_id=0, 
                    
                    model_file = 'model_best.pth.tar'):   
    
    fname, results_dir_root = get_path(set_type)
    
    model_path = os.path.join(results_dir_root, model_path, model_file)
    
    bn = model_path.split(os.sep)[-2]
    
    parts = bn.split('_')
    
    model_name = parts[4] 
    
    if torch.cuda.is_available():
        dev_str = "cuda:" + str(cuda_id)
        print("THIS IS CUDA!!!!")
    else:
        dev_str = 'cpu'

    print(dev_str)
    device = torch.device(dev_str)

    return_label = False
    return_snp = False
    unsampled_test = True
    is_divergent_set = 'divergent_set' in model_path

    

    gen = SkelTrainer(fname = fname,
                      is_divergent_set = is_divergent_set,
                      return_label = return_label,
                      return_snp = return_snp,
                      unsampled_test = unsampled_test)
    
    
    
    model = get_model(model_name, len(gen._strain_dict), gen.embedding_size)
    model = model.to(device)
    
    #assert set_type in model_path
    state = torch.load(model_path, map_location = dev_str)
    print(state['epoch'])
    model.load_state_dict(state['state_dict'])
    model.eval()

    
    inv_strain_dict = {x:k for k,x in gen._strain_dict.items()}
    
    #model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
    
    gen.test()
    test_indexes = gen.valid_index
    
    all_res = []
    with torch.no_grad():
        pbar = tqdm.tqdm(test_indexes)
        for vid_id in pbar:
            x_in = gen._get_data(vid_id)[None, None, :, :]
            
            X = torch.from_numpy(x_in).to(device)
            
            try:
                pred = model(X)
            except:
                continue
            pred = pred.cpu().detach().numpy()[0]
            
            pred_v = np.argmax(pred)
            
            file_id = gen.video_info.loc[vid_id, 'file_id']
            file_path = gen.video_info.loc[vid_id, 'file_path']
            
            
            dd = (file_id, file_path, pred_v, inv_strain_dict[pred_v])
            all_res.append(dd)
    
    df = pd.DataFrame(all_res, columns=['file_id', 'file_path', 'pred_id', 'pred_label'])
     
    return df

if __name__ == '__main__':
   cuda_id = 2

   all_args = [
       ('pesticides resnet18',
         'pesticides-test_angles',
         'logs/pesticides-training_angles_20190409_003715_resnet18_sgd_lr0.0001_wd0.0001_batch8'
         ),
       ('pesticides densenet121',
         'pesticides-test_angles',
         'logs/pesticides-training_angles_20190409_003749_densenet121_sgd_lr0.0001_wd0.0001_batch3'
         ),
       ('pesticides simpledilated',
         'pesticides-test_angles',
         'logs/pesticides-training_angles_20190409_104945_simpledilated_sgd_lr0.001_wd0.0001_batch8'
         )
       ]

   _, results_dir_root = get_path('')
   save_dir = Path(results_dir_root) / 'summary'
   save_dir.mkdir(parents=True, exist_ok=True)
   
   all_summaries = []
   date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   for args in all_args:
       print(args)
       df = get_predictions(*args, cuda_id=cuda_id)
       df.to_csv(save_dir / (date_str + '_' + args[0] + '.csv'))
   
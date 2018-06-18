#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:31:12 2018

@author: avelinojaver
"""
import cv2
import numpy as np
import os
import torch
from path import get_path

from models import SimpleDilated

from flow import collate_fn, SkelTrainer
import tqdm
from torch.utils.data import DataLoader

from train import get_predictions

from torch.nn import functional as F

from sklearn.metrics import f1_score
if __name__ == '__main__':
    set_type = 'angles'
    #set_type = 'AE_emb_20180206'
    
    fname, results_dir_root = get_path(set_type)
    
    model_path = os.path.join(results_dir_root, 'logs/angles_20180531_125503_R_simpledilated_sgd_lr0.0001_wd0_batch8/model_best.pth.tar')
    #model_path = os.path.join(results_dir_root, 'log_divergent_set/angles_20180524_115242_simpledilated_div_lr0.0001_batch8/model_best.pth.tar')
    
    #model_path = os.path.join(results_dir_root, 'log_divergent_set_snp/angles_20180524_222349_simpledilated_div_lr0.0001_batch8/model_best.pth.tar')
    #model_path = os.path.join(results_dir_root, 'logs_snp/angles_20180524_222950_simpledilated_lr0.0001_batch8/model_best.pth.tar')
    
    cuda_id = 0
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
    is_divergent_set = False
    
    gen = SkelTrainer(fname = fname,
                      is_divergent_set = is_divergent_set,
                      return_label = return_label, 
                      return_snp = return_snp,
                      unsampled_test = unsampled_test)
    
    
    model = SimpleDilated(gen.num_classes)
    model = model.to(device)
    
    assert set_type in model_path
    state = torch.load(model_path, map_location = dev_str)
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    #aw = np.load(model_path.replace('.pth.tar', '.npy'))
    
    #%%
    batch_size = 1 if unsampled_test else 4
    loader = DataLoader(gen, 
                        batch_size = batch_size, 
                        collate_fn = collate_fn,
                        num_workers = 0,
                        )
    gen.test()
    
    all_res = []
    with torch.no_grad():
        all_res = []
        pbar = tqdm.tqdm(loader)
        for x_in, y_in in pbar:
            X = x_in.to(device)
            target =  y_in.to(device)
            
            M = model.cnn_clf(X)
            pred =  model.fc_clf(M)
            pred_s = F.softmax(pred, dim=1)
            
            _, res = pred_s.max(1)

            all_res.append((y_in.item(), res.item()))
            #print('T:', y_in, 'R:', )
    
    target, predictions = np.array(all_res).T
    acc = (target == predictions).mean()
    print('acc : {}'.format(acc))

    f1 = f1_score(target, predictions, average='macro')
    print('f1 : {}'.format(f1))

    # with torch.no_grad():
    #     ind = 1
    #     ss = gen._strain_dict[gen.video_info.loc[ind, 'strain']]
    #     print('real' , ss)
        
    #     Xf = torch.from_numpy(gen._full_video(ind).T[None, None])
    #     Xf = Xf.to(device)
    #     print('F:', model(Xf).max(1))
        
    #     for _ in range(3):
            
    #         Xs = torch.from_numpy(gen._sample_video(ind).T[None, None])
    #         Xs = Xs.to(device)
    #         print('S:', model(Xs).max(1))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:31:12 2018

@author: avelinojaver
"""
import cv2
import numpy as np
import os
import pickle
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
    
    bn = model_path.split(os.sep)[-2]
    save_path = os.path.join(results_dir_root, 'maps_clf', bn)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
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
    
    gen.test()
    test_indexes = gen.valid_index

    
    all_res = []
    all_maps = []
    with torch.no_grad():
        all_res = []
        pbar = tqdm.tqdm(test_indexes)
        for vid_id in pbar:
            strain = gen.video_info.loc[vid_id, 'strain']
            target = gen._strain_dict[strain]

            x_in = gen._get_data(vid_id)[None, None, :, :]
            X = torch.from_numpy(x_in).to(device)
            
            M = model.cnn_clf(X)
            pred =  model.fc_clf(M)
            pred_s = F.softmax(pred, dim=1)
            
            _, res = pred_s.max(1)

            all_res.append((target, res.item()))
            
            maps = M.detach().cpu().numpy()
            
            FC = model.fc_clf[2]
            W = FC.weight[target, :]
            B = FC.bias[target]
            
            maps_r = (M*W.view(1, -1, 1, 1) + B).sum(1)
            bot = maps_r.min()
            top = maps_r.max()
            
            maps_r = (maps_r-bot)/(top - bot)
            map_loc = maps_r.detach().cpu().numpy()

            all_maps.append(map_loc)

    targets, predictions = np.array(all_res).T
    acc = (targets == predictions).mean()
    print('acc : {}'.format(acc))

    f1 = f1_score(targets, predictions, average='macro')
    print('f1 : {}'.format(f1))

    fname = os.path.join(save_path, 'maps.p')
    with open(fname, 'bw') as fid:
        data = {'indexes':test_indexes, 
                'predictions':predictions, 
                'targets':targets, 
                'maps':all_maps}
        pickle.dump(data, fid)



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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:19:19 2018

@author: avelinojaver
"""
import sys
sys.path.append('../../../worm-ts-classification')

from worm_ts_classification.flow import SkelTrainer, collate_fn, DIVERGENT_SET
from worm_ts_classification.path import get_path

from torch.utils.data import DataLoader
import tqdm
#%%
if __name__ == '__main__':
    
    #emb_set = 'AE_emb32_20180613_l1'
    #emb_set = 'skeletons'
    #emb_set = 'eigen'
    #emb_set = 'SWDB_angles'
    #emb_set = 'CeNDRAgg_angles'
    #emb_set = 'angles'
    emb_set = 'pesticides-training_angles'
    #emb_set = 'pesticides-test_angles'
    
    fname, results_dir_root = get_path(emb_set)
    
    gen = SkelTrainer(fname = fname, 
                      is_divergent_set = False, 
                      is_tiny = False,
                        return_label = True, 
                        return_snp = False,
                        unsampled_test = False,
                        is_common_WT = False,
                        merge_by_week = False
                        )
    
    #print([gen._strain_dict[x] for x in DIVERGENT_SET])
    #%%
#    fname, results_dir_root = get_path('angles')
#    gen_v2 = SkelTrainer(fname = fname, 
#                      is_divergent_set = False, 
#                      is_tiny = False,
#                        return_label = False, 
#                        return_snp = False,
#                        unsampled_test = True,
#                        sample_size = 22500,
#                        last_frame = 22500
#                        )
    #%%
    gen.test()
    loader = DataLoader(gen, 
                        batch_size = 2, 
                        collate_fn = collate_fn,
                        num_workers = 2
                        )
    
    for X,Y in loader:
        break
#    for X, in gen:
#        break
    
    #%%
    
    dat = []
    
#    #print(gen.video_info.loc[gen.valid_index, 'strain'].value_counts())
#    #for ii, D in enumerate(tqdm.tqdm(gen)):
#    #    dat.append(D[1])
#    
#    for ii, (D, _) in enumerate(gen):    
#        print([x.shape for x in D])
#        
#    
#        plt.figure()
#        plt.imshow(D[0].squeeze(), aspect='auto')
#        
#        if ii > 3:
#            break
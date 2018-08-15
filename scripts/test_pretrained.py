#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:19:19 2018

@author: avelinojaver
"""
import sys
sys.path.append('../../worm-ts-classification')

from worm_ts_classification.path import get_path
from worm_ts_classification.flow import SkelTrainer, collate_fn
from worm_ts_classification.models import ResNetP, SqueezeNetP, VGGP
from worm_ts_classification.models.pretrained_models import _vggs

#%%
from torch.utils.data import DataLoader
import tqdm


if __name__ == '__main__':
    
    #emb_set = 'AE_emb32_20180613_l1'
    #emb_set = 'skeletons'
    #emb_set = 'eigen'
    emb_set = 'SWDB_angles'
    #emb_set = 'SWDB_eigen'
    fname, results_dir_root = get_path(emb_set)
    
    gen = SkelTrainer(fname = fname, 
                      is_divergent_set = False, 
                      is_tiny = False,
                        return_label = True, 
                        return_snp = False,
                        unsampled_test = True,
                        sample_size = 22500,
                        is_only_WT = True
                        )
    
    loader = DataLoader(gen, 
                        batch_size = 2, 
                        collate_fn = collate_fn,
                        num_workers = 2
                        )
    
    
    for X,y in tqdm.tqdm(loader):
        break
    #%%
    #mod = _vggs['vgg11_bn']()
    #dd = mod.features(X.repeat(1,3,1,1))
    
    mod_vgg = VGGP('vgg11bn', gen.num_classes)
    pred = mod_vgg(X)
    
    #%%
    mod_squeezenet = SqueezeNetP('squeezenet11', gen.num_classes)
    pred = mod_squeezenet(X)
    #%%
    mod_resnet = ResNetP('resnet18', gen.num_classes)
    pred = mod_resnet(X)
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:59:17 2018

@author: avelinojaver
"""
import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from worm_ts_classification.flow import SkelTrainer, collate_fn, DIVERGENT_SET
from worm_ts_classification.path import get_path

#%%

import pandas as pd

if __name__ == '__main__':
    save_dir = '/Users/avelinojaver/workspace/WormData/experiments/classify_strains/results/ow_comparison'
    save_dir = Path(save_dir)
    
    df = pd.read_csv('/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_features_experiments/classify/data/SWDB/ow_features_full_SWDB.csv')
    
    emb_set = "SWDB_angles"
    fname, results_dir_root = get_path(emb_set)
    
    gen = SkelTrainer(fname = fname, 
                      is_divergent_set = False, 
                      is_tiny = False,
                        return_label = False, 
                        return_snp = False,
                        unsampled_test = True,
                        sample_size = 22500
                        )
    video_info = gen.video_info
    #%%
    cols2ignore = ['id',
 'date',
 'original_video',
 'original_video_sizeMB',
 'directory',
 'strain',
 'strain_description',
 'allele',
 'gene',
 'chromosome',
 'tracker',
 'sex',
 'developmental_stage',
 'ventral_side',
 'food',
 'habituation',
 'experimenter',
 'arena',
 'exit_flag',
 'experiment_id',
 'n_valid_frames',
 'n_missing_frames',
 'n_segmented_skeletons',
 'n_filtered_skeletons',
 'n_valid_skeletons',
 'n_timestamps',
 'first_skel_frame',
 'last_skel_frame',
 'fps',
 'total_time',
 'microns_per_pixel',
 'mask_file_sizeMB',
 'skel_file',
 'frac_valid']
    
    
    
 
    infocols = ['id',
 'worm_id',
 'base_name',
 'date',
 'results_dir',
 'strain',
 'strain_description',
 'allele',
 'gene',
 'chromosome',
 'tracker',
 'sex',
 'developmental_stage',
 'days_of_adulthood',
 'ventral_side',
 'food',
 'habituation',
 'experimenter',
 'arena',
 'exit_flag',
 'n_valid_frames',
 'n_missing_frames',
 'n_valid_skeletons',
 'fps',
 'total_time',
 'microns_per_pixel',
 'file_path',
 'fold']
    
    cols = [x for x in df if x not in cols2ignore]
    df_filt = df.loc[df['base_name'].isin(video_info['base_name']), cols]
    df_merged = pd.merge(video_info, df_filt, on='base_name')
    
    feat_cols = [x for x in df_merged if x not in infocols]
    df_info = df_merged.loc[:, infocols]
    df_info['strain_id'] = df_info['strain'].map(gen._strain_dict)
    
    
    df_feats = df_merged.loc[:, feat_cols]
    
    df_feats_z = (df_feats - df_feats.mean())/df_feats.std()
    df_feats_z[df_feats_z.isnull()] = 0
    
    
    df_info.to_csv(save_dir / 'SWDB_Y_info.csv', index=False)
    df_feats_z.to_csv(save_dir / 'SWDB_X_features.csv', index=False)
    
    #%%
    df = pd.read_csv('/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_features_experiments/classify/data/CeNDR/ow_features_full_CeNDR.csv')
    
    emb_set = "angles"
    fname, results_dir_root = get_path(emb_set)
    
    gen = SkelTrainer(fname = fname, 
                      is_divergent_set = True, 
                      is_tiny = False,
                        return_label = False, 
                        return_snp = False,
                        unsampled_test = True,
                        sample_size = 22500
                        )
    video_info = gen.video_info
    #%%
    cols = [x for x in df if x not in cols2ignore]
    
    video_info['base_name'] = video_info['base_name'].str.lower()
    df_filt = df.loc[df['base_name'].isin(video_info['base_name']), cols]
    df_merged = pd.merge(video_info, df_filt, on='base_name')
    #%%
    
    infocols = ['strain',
 'set_n',
 'n_worms',
 'date',
 'time',
 'file_path',
 'base_name',
 'fold',
 'id.1',
 'exp_name']
    
    cols = [x for x in df if x not in cols2ignore]
    df_filt = df.loc[df['base_name'].isin(video_info['base_name']), cols]
    df_merged = pd.merge(video_info, df_filt, on='base_name')
    
    feat_cols = [x for x in df_merged if x not in infocols]
    df_info = df_merged.loc[:, infocols]
    df_info['strain_id'] = df_info['strain'].map(gen._strain_dict)
    
    
    df_feats = df_merged.loc[:, feat_cols]
    
    df_feats_z = (df_feats - df_feats.mean())/df_feats.std()
    df_feats_z[df_feats_z.isnull()] = 0
    
    df_info.to_csv(save_dir / 'CeNDR_Y_info.csv', index=False)
    df_feats_z.to_csv(save_dir / 'CeNDR_X_features.csv', index=False)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:44:52 2018

@author: avelinojaver
"""
import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))


from worm_ts_classification.flow import SkelTrainer

import pandas as pd
#%%
if __name__ == '__main__':
    set_type = 'CeNDRAgg_angles'
    fname, results_dir_root = get_path(set_type)
    gen = SkelTrainer(fname = fname,
                      is_divergent_set = False,
                      return_label = True,
                      return_snp = False,
                      unsampled_test = True,
                      train_epoch_magnifier=1
                      )

#%%

#src_file = Path.home() / 'OneDrive - Nexus365/aggregation'
#feat_file =  src_file / 'features_summary_tierpsy_plate_20180910_165259.csv'
#fnames_file = src_file / 'filenames_summary_tierpsy_plate_20180910_165259.csv'
#
#df_feats = pd.read_csv(feat_file)
#df_files = pd.read_csv(fnames_file)
#
##%%
#plt.plot(df_feats['blob_area_90th'].sort_values().values, '.')
#

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 13:04:37 2018

@author: avelinojaver
"""
from pathlib import Path
import sys

src_d = Path(__file__).resolve().parents[1]
sys.path.append(str(src_d))

from worm_ts_classification.path import _root_dirs
from embeddings_helper import calculate_embeddings

import numpy as np
import glob
import datetime
import pandas as pd
import os

#%%
#def get_video_info_from_files(fnames, root_dir, f_ext = '_featuresN.hdf5'):
#    
#    dd = []
#    for fname in fnames:
#        bn = os.path.basename(fname).replace(f_ext, '')
#        
#        parts = bn.split('_')
#        parts = parts[:-1] if parts[-1] == 'ROIs' else parts
#        
#        
#        strain = parts[0]
#        n_worms = int(parts[1][5:])
#        
#        dt =datetime.datetime.strptime(parts[-2] + parts[-1], '%d%m%Y%H%M%S')
#        date_str = datetime.datetime.strftime(dt, '%Y-%m-%d')
#        time_str = datetime.datetime.strftime(dt, '%H:%M:%S')
#        
#        fname_r = str(fname).replace(str(root_dir), '')
#        
#        set_n = int(fname_r.partition('_Set')[-1][0])
#        
#        row = (strain, set_n, n_worms, date_str, time_str, fname_r)
#        dd.append(row)
#    
#    video_info = pd.DataFrame(video_info, columns=['strain', 'set_n', 'n_worms', 'date', 'time', 'file_path'])
#    return video_info, fnames

if __name__ == '__main__':
    p = 'osx' if sys.platform == 'darwin' else 'centos_oxford'
    root = _root_dirs[p]

    emb_set = 'angles'
    
    #set_type = 'CeNDR'
    #root_dir = root + 'screenings/CeNDR/Results'
    
    for partition_type in ['training']:#['test', 'training']:
    
        set_type = f'pesticides-{partition_type}'
        
        root_dir = root + f'screenings/pesticides/{partition_type}'
        root_dir = Path(root_dir)
        assert root_dir.exists()
        
        
        csv_filenames = root_dir / f'{partition_type}_featuresN_filenames.csv'
        
        video_info = pd.read_csv(csv_filenames)
        video_info.columns = ['file_id', 'file_path']
        video_info['file_path'] = video_info['file_path'].str.replace(f'./{partition_type}_set_featuresN/', '')
        
        labels_file = root_dir / f'{partition_type}_labels.csv'
        if labels_file.exists():
            df_labels = pd.read_csv(labels_file)
            df_labels = df_labels[['MOA_group', 'MOA_general', 'MOA_specific']]
            df_labels['MOA_group'] = df_labels['MOA_group'].astype(np.int)
            video_info = video_info.join(df_labels, on = 'file_id')
            
        
        fnames = [root_dir / x for x in video_info['file_path']]
        
        
        save_file = root + 'experiments/classify_strains/{}_{}.hdf5'.format(set_type, emb_set)
        
        calculate_embeddings(video_info, 
                             fnames, 
                             emb_set,
                             save_file)
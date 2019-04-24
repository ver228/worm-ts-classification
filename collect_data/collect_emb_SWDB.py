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

import glob
import datetime
import pandas as pd
import os


#%%

if __name__ == '__main__':
    p = 'osx' if sys.platform == 'darwin' else 'centos_oxford'
    root = _root_dirs[p]

    emb_sets = ['angles', 'skeletons', 'eigen', 'eigenfull']
    
    set_type = 'SWDB'
    root_dir = str(Path.home() / 'workspace' / 'WormData' / 'screenings')
    f_ext = '_interpolated25.hdf5'
    
    
    
    old_root = '/Volumes/behavgenom_archive\$/'
    csv_file = Path(__file__).resolve().parents[2] / 'single_worm' / 'interpolated_skeletons' / 'strains2process.csv'
        
    video_info = pd.read_csv(str(csv_file))
    video_info['file_path'] = video_info['results_dir'].str.replace(old_root, '') + os.sep + video_info['base_name']
    fnames = [str(Path(root_dir) / (x['file_path'] + f_ext)) for _, x in video_info.iterrows()]
        
    for emb_set in emb_sets:
        save_file = root + 'experiments/classify_strains/{}_{}.hdf5'.format(set_type, emb_set)
        calculate_embeddings(video_info, 
                             fnames, 
                             emb_set,
                             save_file)
    
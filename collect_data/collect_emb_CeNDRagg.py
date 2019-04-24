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

import pandas as pd
import os


def get_video_info_from_csv_agg(root_dir):
    csv_path = str(Path.home() / 'workspace/WormData/screenings/Serena_WT_Screening/metadata_aggregation_screening.csv')
    bad_labels = ['NONE']
    
    video_info = pd.read_csv(csv_path)
    video_info['dirname'] = video_info['dirname'].str.replace('/Volumes/behavgenom_archive\$/Serena/AggregationScreening/MaskedVideos/', '')
    video_info = video_info.rename(columns={'strain_name':'strain', 'dirname':'file_path'})
    video_info = video_info[~video_info['strain'].isin(bad_labels)]
    

    fnames = root_dir + '/' + video_info['file_path'] + '/'+  video_info['basename'].str.replace('.hdf5', '_featuresN.hdf5')
    is_valid = [os.path.exists(x) for x in fnames.values]

    video_info = video_info[is_valid]
    fnames = [Path(x) for e, x in zip(is_valid,fnames.values) if e]

    return video_info, fnames

#%%

if __name__ == '__main__':
    p = 'osx' if sys.platform == 'darwin' else 'centos_oxford'
    root = _root_dirs[p]

    emb_sets = ['angles']
    
    set_type = 'CeNDRAgg'
    root_dir = root + 'screenings/Serena_WT_Screening/Results'
    
    
    video_info, fnames = get_video_info_from_csv_agg(root_dir)
    for emb_set in emb_sets:
        save_file = root + 'experiments/classify_strains/{}_{}.hdf5'.format(set_type, emb_set)
        calculate_embeddings(video_info, 
                             fnames, 
                             emb_set,
                             save_file)
    
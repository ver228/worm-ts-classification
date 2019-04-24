#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:34:58 2018

@author: avelinojaver
"""

from pathlib import Path
import sys

src_d = Path(__file__).resolve().parents[2] / 'worm_ts_classification'
sys.path.append(str(src_d))

from path import _root_dirs

from tierpsy.summary.collect import calculate_summaries


if __name__ == '__main__':
    p = 'osx' if sys.platform == 'darwin' else 'centos_oxford'
    root = _root_dirs[p]
    
    root_dir = str(root + 'screenings/Serena_WT_Screening/Results')
    
    feature_type = 'tierpsy'
    summary_type = 'plate'
    is_manual_index = False
    
    df_files, all_summaries = calculate_summaries(root_dir, 
                                                  feature_type, 
                                                  summary_type, 
                                                  is_manual_index, 
                                                  _is_debug = False
                                                  )
    
    import pdb
    pdb.set_trace()
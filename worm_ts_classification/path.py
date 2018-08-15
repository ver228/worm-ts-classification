#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:17:29 2018

@author: avelinojaver
"""
import os
import sys

_root_dirs = {
        'osx' : '/Volumes/rescomp1/data/WormData/',
        #'osx' : '/Users/avelinojaver/Data/',
        'centos_oxford' : '/well/rittscher/users/avelino/WormData/',
        'loc' : '/Users/avelinojaver/Documents/Data/',
        'tmp' : '/tmp/'
        }

def get_base_dir(platform=None):
    if platform is None:
        platform = 'osx' if sys.platform == 'darwin' else 'centos_oxford'
    
    root = _root_dirs[platform]
    base_dir = root + 'experiments/classify_strains/'
    
    return base_dir

def get_path(emb_set, platform = None, is_tmp = True):
    
    base_dir = get_base_dir(platform)
    
    dname = base_dir + 'data/'
    fname = dname + '{}.hdf5'.format(emb_set)
    if not os.path.exists(fname):
        fname = fname = dname + 'CeNDR_{}.hdf5'.format(emb_set)
    
    
    results_dir = os.path.join(base_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    return fname, results_dir

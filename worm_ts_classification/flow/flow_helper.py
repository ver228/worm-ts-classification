#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 16:01:29 2018

@author: avelinojaver
"""
import os
import itertools
import pickle
import numpy as np
import pandas as pd

from pathlib import Path


ROOT_DIR = os.path.dirname(__file__)
DFLT_SNP_FILE = os.path.join(ROOT_DIR, 'CeNDR_snps.csv')

def get_folds_file(fname):
    bn = Path(fname).name
    if bn.startswith('SWDB'):
        folds_file = os.path.join(ROOT_DIR, 'SWDB_fold_dict.p')
    elif bn.startswith('CeNDR'):
        folds_file = os.path.join(ROOT_DIR, 'CeNDR_fold_dict.p')
    else:
        raise ValueError
    return folds_file

def read_CeNDR_snps(source_file = DFLT_SNP_FILE):
    snps = pd.read_csv(source_file)
    
    info_cols = snps.columns[:4]
    strain_cols = snps.columns[4:]
    snps_vec = snps[strain_cols].copy()
    snps_vec[snps_vec.isnull()] = 0
    snps_vec = snps_vec.astype(np.int8)
    
    
    snps_c = snps[info_cols].join(snps_vec)
    
    r_dtype = []
    for col in snps_c:
        dat = snps_c[col]
        if dat.dtype == np.dtype('O'):
            n_s = dat.str.len().max()
            dt = np.dtype('S%i' % n_s)
        else:
            dt = dat.dtype
        r_dtype.append((col, dt))
    
    snps_r = snps_c.to_records(index=False).astype(r_dtype)
    snps_r = pd.DataFrame(snps)
    
    return snps_r

def get_strains_ids(snps):
    valid_strains = snps.columns[4:].tolist()
    strain_dict = {k:ii for ii, k in enumerate(valid_strains)}
    return strain_dict


def add_bn_series(df):
    def _remove_end(x, postfix):
        return x[:-len(postfix)] if x.endswith(postfix) else x 
        
    def _get_base_name(x):
        bn = os.path.basename(x)
        bn = _remove_end(bn, '_featuresN.hdf5')
        bn = _remove_end(bn, '_embeddings.hdf5')
        bn = _remove_end(bn, '_ROIs')
        return bn
    
    df['base_name'] = df['file_path'].apply(_get_base_name)
    return df


def save_folds_dict(df, source_file, seed = 777):
    df = add_bn_series(df)
    folds_dict = {}
    for ss, dat in df.groupby('strain'):
        gen = itertools.cycle(range(3))
        for bn in dat['base_name'].values:
            folds_dict[bn] = next(gen)
    
    save_file = get_folds_file(source_file)
    with open(save_file, 'wb') as fid:
        pickle.dump(folds_dict, fid)

def add_folds(df, source_file):
    folds_file = get_folds_file(source_file)
    df = add_bn_series(df)
    with open(folds_file, 'rb') as fid:
        folds_dict = pickle.load(fid)
    df['fold'] = df['base_name'].map(folds_dict)
    
    if np.any(np.isnan(df['fold'])):
        import pdb
        pdb.set_trace()
        raise ValueError()
    
    return df
#%%
def _get_strain_dict_file(source_file):
    bn = Path(source_file).name.partition('_')[0]
    strain_dict_file = os.path.join(ROOT_DIR, bn + '_straindict.p')
    return strain_dict_file

def save_strain_dict(df, source_file):
    strain_dict_file = _get_strain_dict_file(source_file)
    strain_dict = {x:ii for ii,x in enumerate(sorted(df['strain'].unique()))}
    with open(strain_dict_file, 'wb') as fid:
        pickle.dump(strain_dict, fid)
        
def load_strain_dict(source_file):
    strain_dict_file = _get_strain_dict_file(source_file)
    with open(strain_dict_file, 'rb') as fid:
        strain_dict = pickle.load(fid)
    return strain_dict
#%%
if __name__ == '__main__':
    #fname = '/Users/avelinojaver/Documents/Data/experiments/classify_strains/CeNDR_angles.hdf5'
    fname = Path.home() / 'workspace/WormData/experiments/classify_strains/SWDB_angles.hdf5'
    with pd.HDFStore(fname) as fid:
        video_info = fid['/video_info']
        video_info['strain'] = video_info['strain'].str.strip(' ')
    
    #save_folds_dict(video_info, fname, seed = 777)
        
    video_info = add_folds(video_info, fname)
    #%%
    
    
    
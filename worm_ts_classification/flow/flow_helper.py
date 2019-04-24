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

import warnings
from pathlib import Path


ROOT_DIR = os.path.dirname(__file__)
DFLT_SNP_FILE = os.path.join(ROOT_DIR, 'CeNDR_snps.csv')

DIVERGENT_SET = ['N2',
 'CB4856',
 'DL238',
 'JU775',
 'MY16',
 'MY23',
 'CX11314',
 'ED3017',
 'EG4725',
 'LKC34',
 'JT11398',
 'JU258']

SWDB_STRAINS = dict(
    N2_types = ['AQ2947', 'AQ3000', 'N2'],
    
    wt_isolates = ['CB4852','CB4853','CB4856','ED3017','ED3021','ED3049',
                   'ED3054','JU258','JU298','JU343','JU345','JU393','JU394',
                   'JU402','JU438','JU440','LSJ1','MY16','PS312','RC301'
                   ],
                   
    mutants = ['AQ1031','AQ1033','AQ1037','AQ1038','AQ1413','AQ1422', 
                  'AQ1441','AQ2056','AQ2153','AQ2197','AQ2316','AQ2533', 
                  'AQ2649','AQ2932','AQ2933','AQ2934','AQ2935','AQ2936', 
                  'AQ2937','AQ495','AQ866','AQ908','AQ916','AX1410','AX1743', 
                  'AX1745','BA1093','BR4006','BZ28','CB101','CB102','CB1068',
                  'CB1069','CB109','CB1112','CB1141','CB1197','CB120','CB1265',
                  'CB1282','CB1313','CB1376','CB1386','CB1416','CB1460','CB15',
                  'CB151','CB1515','CB1562','CB1597','CB1598','CB169','CB189',
                  'CB193','CB262','CB270','CB3203','CB402','CB4371','CB4870',
                  'CB491','CB5','CB502','CB566','CB57','CB587','CB678','CB723',
                  'CB755','CB81','CB845','CB904','CB933','CB94','CB950','CE1047',
                  'CF263','CX10','CX2205','CX6391','DA1674','DA1814','DA2100',
                  'DA609','DG1856','DH244','DR1','DR1089','DR2','DR62','DR96', 
                  'EG106','EJ26','EU1006','EW35','FF41','FG58','FX1053',
                  'FX1498','FX1553','FX1568','FX1583','FX1846','FX1880','FX1908',
                  'FX2146','FX2173','FX2427','FX2706','FX3085','FX3092','FX863',
                  'HE130','HH27','IC683','IK130','JC55','JR2370','JT603','JT609',
                  'KG1180','KG421','KP2018','KS99','KU25','LX533','LX636','LX702',
                  'LX703','LX704','LX950','LX981','LX982','MP145','MT1067',
                  'MT1073','MT1078','MT1079','MT1081','MT1082','MT1083','MT1093',
                  'MT1179','MT1200','MT1202','MT1205','MT1216','MT1217','MT1222',
                  'MT1231','MT1232','MT1236','MT1241','MT13113','MT13292','MT1444',
                  'MT151','MT1514','MT1540','MT1543','MT15434','MT155','MT1656',
                  'MT180','MT2068','MT2246','MT2247','MT2248','MT2293','MT2316',
                  'MT2611','MT324','MT6129','MT7988','MT8504','MT8944','MT9455',
                  'MT9668','NC279','NL1137','NL1142','NL1146','NL1147','NL2330',
                  'NL332','NL334','NL335','NL594','NL787','NL790','NL792','NL793',
                  'NL795','NL797','NM1657','NM210','NY106','NY119','NY133',
                  'NY183','NY184','NY193','NY2099','NY227','NY228','NY230',
                  'NY244','NY245','NY247','NY248','NY249','NY32','NY34','NY7',
                  'OH313','PS4330','QL127','QL13','QL131','QL14','QL15','QL16',
                  'QL17','QL18','QL188','QL189','QL19','QL20','QL21','QL22','QL23',
                  'QL24','QL26','QL27','QL28','QL47','QL48','QL49','QL50','QL51',
                  'QL53','QL76','QT309','QZ104','QZ126','QZ80','QZ81','RB1030',
                  'RB1052','RB1064','RB1132','RB1156','RB1172','RB1177','RB1192',
                  'RB1226','RB1250','RB1263','RB1316','RB1340','RB1350','RB1356',
                  'RB1372','RB1374','RB1380','RB1396','RB1523','RB1543','RB1546',
                  'RB1559','RB1609','RB1659','RB1800','RB1802','RB1809','RB1816',
                  'RB1818','RB1832','RB1863','RB1883','RB1902','RB1911','RB1915',
                  'RB1958','RB1989','RB1990','RB2005','RB2030','RB2059','RB2067',
                  'RB2098','RB2119','RB2126','RB2159','RB2188','RB2262','RB2275',
                  'RB2294','RB2351','RB2412','RB2489','RB2498','RB2544','RB2552',
                  'RB2575','RB2594','RB2615','RB503','RB557','RB559','RB607',
                  'RB641','RB648','RB668','RB680','RB687','RB709','RB738','RB753',
                  'RB756','RB761','RB777','RB799','RB873','RB905','RB919','RB946',
                  'RB982','RM2702','RM2710','RW85','SP1789','TG34','TQ194',
                  'TQ225','TQ296','VC1024','VC1063','VC12','VC1218','VC1243',
                  'VC125','VC1295','VC1309','VC1340','VC1528','VC1759','VC1982',
                  'VC223','VC224','VC2324','VC2423','VC2497','VC2502','VC282',
                  'VC335','VC602','VC649','VC731','VC8','VC840','VC854','VC9',
                  'VC975','VP91','ZZ15','ZZ427'
                  ]
)

STRAINS_IN_COMMON = ['N2', 'CB4856', 'ED3017', 'JU258', 'MY16']

def get_folds_file(fname):
    bn = Path(fname).name
    if bn.startswith('SWDB'):
        folds_file = os.path.join(ROOT_DIR, 'SWDB_fold_dict.p')
    elif bn.startswith('CeNDRAgg'):
        folds_file = os.path.join(ROOT_DIR, 'CeNDRAgg_fold_dict.p')
    elif bn.startswith('CeNDR'):
        folds_file = os.path.join(ROOT_DIR, 'CeNDR_fold_dict.p')
    elif bn.startswith('pesticides-training'):
        folds_file = os.path.join(ROOT_DIR, 'pesticides-training_fold_dict.p')
    else:
        warnings.warn('Folds are not added since {} does not exists.'.format(bn))
        folds_file = None
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


def save_folds_dict(df, source_file, field2group = 'strain', n_folds = 3, seed = 777):
    df = add_bn_series(df)
    folds_dict = {}
    for ss, dat in df.groupby(field2group):
        gen = itertools.cycle(range(n_folds))
        for bn in dat['base_name'].values:
            folds_dict[bn] = next(gen)
    
    save_file = get_folds_file(source_file)
    with open(save_file, 'wb') as fid:
        pickle.dump(folds_dict, fid)

def add_folds(df, source_file):
    folds_file = get_folds_file(source_file)
    if folds_file is None:
        return df
    
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
    bn = bn.partition('-')[0]
    strain_dict_file = os.path.join(ROOT_DIR, bn + '_straindict.p')
    return strain_dict_file

def save_strain_dict(df, source_file, field2group = 'strain'):
    strain_dict_file = _get_strain_dict_file(source_file)
    strain_dict = {x:ii for ii,x in enumerate(sorted(df[field2group].unique()))}
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
    #fname = Path.home() / 'workspace/WormData/experiments/classify_strains/SWDB_angles.hdf5'
    #fname = Path.home() / 'workspace/WormData/experiments/classify_strains/data/CeNDRAgg_angles.hdf5'
    fname = Path.home() / 'workspace/WormData/experiments/classify_strains/data/pesticides-training_angles.hdf5'
    
    with pd.HDFStore(fname) as fid:
        video_info = fid['/video_info']
#        video_info['strain'] = video_info['strain'].str.strip(' ')
    field2group = 'MOA_group'
    n_folds = 10
    save_folds_dict(video_info, 
                    fname, 
                    field2group = field2group, 
                    n_folds = n_folds,
                    seed = 777)
    
    video_info = add_folds(video_info, fname)
    
    save_strain_dict(video_info, fname, field2group = field2group)
    #%%
    
    
    
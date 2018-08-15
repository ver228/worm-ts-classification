#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:08:14 2018

@author: avelinojaver
"""
from pathlib import Path
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix
#%%
if __name__ == '__main__':
    root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/papers/eccv2018/figures/output_embeddings/')
    #%%
    prediction_files = {
            'SWDB on CeNDR' : 'Base=SWDB-Diff=CeNDR.csv',
            'CeNDR on SWDB' : 'Base=CeNDR-Diff=SWDB.csv',
            'S SWDB on CeNDR' : 'S_Base=SWDB-Diff=CeNDR.csv',
            'S CeNDR on SWDB' : 'S_Base=CeNDR-Diff=SWDB.csv',
            }
    
    
    for d_type, fname in prediction_files.items():
        fname = root_dir / fname
        
        df = pd.read_csv(fname)
        del df['Unnamed: 0']
        
        df = df.applymap(lambda x : x.partition('_')[0])
        
        
        
        y_true = df['target_strain']
        
        fig, axs = plt.subplots(1,2, figsize=(10, 5))
        for ii, xx in enumerate(['predicted_strain_same', 'predicted_strain_diff']):
            y_pred = df[xx]
            df_confusion = pd.crosstab(y_true, y_pred)
            
            valid_strains = df_confusion.index.values
            df_n = df_confusion.div(df_confusion.sum(axis=1), axis=0)
            
            if ii == 0:
                df_n = df_n.loc[:, valid_strains]
            df_n[df_n.isnull()] = 0
            sns.heatmap(df_n, annot=True, cbar=False, ax=axs[ii])
            #(y_true==y_pred).sum()/len(df)
            
        plt.suptitle(d_type)
        
        
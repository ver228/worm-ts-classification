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
    root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/papers/pheno_progress/cross_accuracy/')
    #%%
    prediction_files = {
            ('All', 'SW', 'MWv2') : 'all_Base=SWDB-Diff=CeNDR.csv',
            ('All', 'MWv2', 'SW'): 'all_Base=CeNDR-Diff=SWDB.csv',
            ('5 Strains', 'SW', 'MWv2'): 'small_Base=SWDB-Diff=CeNDR.csv',
            ('5 Strains', 'MWv2', 'SW') : 'small_Base=CeNDR-Diff=SWDB.csv',
            ('resnet18', 'SW', 'MWv2'): 'resnet18_Base=SWDB-Diff=CeNDR.csv',
            ('resnet18', 'MWv2', 'SW') : 'resnet18_Base=CeNDR-Diff=SWDB.csv',
            }
    
    
    for (d_type, s_trained, s_flow), fname in prediction_files.items():
        fname = root_dir / fname
        
        df = pd.read_csv(fname)
        del df['Unnamed: 0']
        
        df = df.applymap(lambda x : x.partition('_')[0])
        
        
        df.columns = [' '.join([x.title() for x in col.split('_')]) for col in df.columns]
        
        
        y_true = df['Target Strain']
        
        
        for ii, col in enumerate(['Predicted Strain Same', 'Predicted Strain Diff']):
            y_pred = df[col]
            df_confusion = pd.crosstab(y_true, y_pred)
            
            valid_strains = df_confusion.index.values
            df_n = df_confusion.div(df_confusion.sum(axis=1), axis=0)
            
            df_n = df_n.loc[:, valid_strains]
            df_n[df_n.isnull()] = 0
            fig, axs = plt.subplots(1,1, figsize=(4,4))
            sns.heatmap(df_n, annot=True, cbar=False, ax=axs, vmin=0, vmax=1)
            
            
            end_s = col.rpartition(' ')[-1]
            if end_s == 'Diff':
                ss = '{}, train:{} source:{}'.format(d_type, s_trained, s_flow)
            else:
                ss = '{}, train:{} source:{}'.format(d_type, s_flow, s_flow)
            
            
            plt.suptitle(ss)
            
            ss = ss.replace(' ', '_').replace(':', '-')
            fig.savefig(ss + '.pdf')
            #(y_true==y_pred).sum()/len(df)
            
        #
        
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:08:23 2018

@author: avelinojaver
"""
#%%
src_file = '/Users/avelinojaver/OneDrive - Nexus365/papers/pheno_progress/tables/Summary_2018-08-23 12:55:39.csv'
df = pd.read_csv(src_file)
df = df.sort_values('name')

ll = []
for _, row in df.iterrows():
    name = row['name'].replace('_', ' ')
    dd = '{} & {:.02f} & {:.02f} & {:.04f} \\\\'.format(name, 100*row['acc_top1'], 100*row['acc_top5'], row['f1'])
    ll.append(dd)
ll = '\n'.join(ll)
print(ll)
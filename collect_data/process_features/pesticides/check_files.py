#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:08:45 2019

@author: avelinojaver
"""
from pathlib import Path
import pandas as pd
import tqdm
import tables


if __name__ == '__main__':
    missing_files = []
    
    
    for t_type in ['test', 'training']:
        root_dir = Path.home() / f'workspace/WormData/screenings/pesticides/{t_type}'
        test_csv = root_dir / f'{t_type}_featuresN_filenames.csv'
        
        df = pd.read_csv(test_csv)
        
        
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=t_type):
            
            try:
                fname = root_dir / row['file_dest'].replace(f'./{t_type}_set_featuresN/', '')
                
                with pd.HDFStore(str(fname)) as fid:
                    fid.get_node('/coordinates/skeletons')[:]
                    df = fid['/trajectories_data']
            except:
                #raise
                missing_files.append(fname)
        
        for fname in missing_files:
            print(str(fname))
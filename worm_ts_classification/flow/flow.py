#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 13:04:37 2018

@author: avelinojaver
"""
from .flow_helper import add_folds, read_CeNDR_snps, load_strain_dict, DIVERGENT_SET, SWDB_STRAINS, STRAINS_IN_COMMON

import pandas as pd
import tables
import numpy as np
import os
import random

import torch
from torch.utils.data import Dataset



def _merge_videos_by_week(video_info, trajectories_ranges):
    video_info['date'] = pd.to_datetime(video_info['date']).dt.normalize()
    video_info['week'] = video_info['date'] - pd.to_timedelta(video_info['date'].dt.dayofweek, unit='d')
    
    cur_vid_ids = 0
    vid_ids_conversion = {}
    new_video_info = []
    for (fold, date, strain), dat in video_info.groupby(['fold', 'week', 'strain']):
        for ind, row in dat.iterrows():
            vid_ids_conversion[ind] = cur_vid_ids
        new_video_info.append((cur_vid_ids, fold, date, strain))
        cur_vid_ids += 1
    
    trajectories_ranges['old_video_id'] = trajectories_ranges['video_id']
    trajectories_ranges['video_id'] = trajectories_ranges['video_id'].map(vid_ids_conversion)
    
    
    
    new_video_info = pd.DataFrame(new_video_info, columns = ['video_id', 'fold', 'week', 'strain'])
    
    
    return new_video_info, trajectories_ranges


class SkelEmbeddingsFlow(Dataset):
    def __init__(self,
                 fname = '',
                 min_traj_size = 250,
                 sample_size = 22500,
                 fold_n_test = 0,
                 train_epoch_magnifier = 5,
                 is_divergent_set = False,
                 is_tiny = False,
                 is_balance_training = False,
                 is_only_WT = False,
                 is_common_WT = False,
                 unsampled_test = False,
                 merge_by_week = False,
                 initial_frame = 0,
                 last_frame = None
                 ):
        
        assert os.path.exists(fname)
        
        self.fname = fname
        self.min_traj_size = min_traj_size
        self.sample_size = sample_size
        self.train_epoch_magnifier = train_epoch_magnifier
        self.is_balance_training = is_balance_training
        self.unsampled_test = unsampled_test
        
        with pd.HDFStore(fname) as fid:
            trajectories_ranges = fid['/trajectories_ranges']
            video_info = fid['/video_info']
            video_info['strain'] = video_info['strain'].str.strip(' ')
            video_info = add_folds(video_info, self.fname)
            
            if 'wormNum' in video_info:
                #hack to include only videos with 5 worms in CeNDRAgg I will need to make this nicer later on
                video_info = video_info[video_info['wormNum']==5]
            
            if last_frame is None:
                last_frame = trajectories_ranges['frame_fin'].max()
            else:
                assert last_frame <= trajectories_ranges['frame_fin'].max()
            assert initial_frame < last_frame
            
            good = (trajectories_ranges['frame_ini'] >= initial_frame) & (trajectories_ranges['frame_ini'] < last_frame-min_traj_size)
            trajectories_ranges = trajectories_ranges[good]
            
            #clip data to be within the given frame ranges
            to_clip = (trajectories_ranges['frame_fin']>last_frame).values
            
            fshifts = int(last_frame)-trajectories_ranges.loc[to_clip, 'frame_fin']
            trajectories_ranges.loc[to_clip, 'row_fin'] += fshifts
            trajectories_ranges.loc[to_clip, 'frame_fin'] = last_frame
            
            #add the size of each chuck in the video
            trajectories_ranges['size'] = trajectories_ranges['frame_fin'] - trajectories_ranges['frame_ini']
            
            #filter chucks that are too small
            trajectories_ranges = trajectories_ranges[trajectories_ranges['size'] >= min_traj_size]
            
            #filter videos that do not have valid trajectories after filtering
            video_info = video_info.loc[trajectories_ranges['video_id'].unique()]
            
            #get the total size of all the chucks in a video
            #tot_ranges = trajectories_ranges.groupby('video_id').agg({'size':'sum'})
            #video_info = video_info[tot_ranges['size'] >= sample_size]
            
            #only keep the strains that have at least 3 valid videos
            vid_per_strain = video_info['strain'].value_counts()
            ss = vid_per_strain.index[vid_per_strain.values > 2]
            video_info = video_info[video_info['strain'].isin(ss)]
        
        if merge_by_week:
            video_info, trajectories_ranges = _merge_videos_by_week(video_info, trajectories_ranges)
        
        with tables.File(fname) as fid:
            skels_g = fid.get_node('/embeddings')
            self.embedding_size = skels_g.shape[1]
        
        if is_tiny:
            good = video_info['strain'].isin(['N2', 'CB4856'])
            video_info = video_info[good]
        if is_divergent_set:
            good = video_info['strain'].isin(DIVERGENT_SET)
            video_info = video_info[good]
            
        if is_only_WT:
            valid_strains = ['N2_XX', 'N2_XO'] + [x + '_XX' for x in SWDB_STRAINS['wt_isolates']]
            
            good = video_info['strain'].isin(valid_strains)
            video_info = video_info[good]
            
        if is_common_WT:
            valid_strains = STRAINS_IN_COMMON
            if not (video_info['strain']=='N2').any():
                valid_strains = [x + '_XX' for x in valid_strains]
            good = video_info['strain'].isin(valid_strains)
            video_info = video_info[good]
            
        
        self.video_info = video_info
        self.video_traj_ranges = trajectories_ranges.groupby('video_id')
        self.fold_n_test = fold_n_test
        
        
        self.train_index = self.video_info.index[self.video_info['fold'] != self.fold_n_test].tolist()
        self.test_index = self.video_info.index[self.video_info['fold'] == self.fold_n_test].tolist()
        
        self.train()
    
    def train(self):
        self.is_train = True
        tot = len(self.train_index)*self.train_epoch_magnifier
        
        #balance classes sampling
        if self.is_balance_training:
            strain_g = self.video_info.loc[self.train_index].groupby('strain').groups
            strains = list(strain_g.keys())
            self.valid_index = []
            while len(self.valid_index) < tot:
                random.shuffle(strains)
                for ss in strains:
                    ind = random.choice(strain_g[ss])
                    self.valid_index.append(ind)
        else:
            self.valid_index = [random.choice(self.train_index) for _ in range(tot)]
            
    def test(self):
        self.is_train = False
        self.valid_index = self.test_index
    
    
    def __getitem__(self, index):
        vid_id = self.valid_index[index]
        return self._get_data(vid_id)
        
    
    def _get_data(self, vid_id):
        if not self.unsampled_test:
            dat = self._sample_video(vid_id)
        else:
            dat = self._full_video(vid_id)
        return dat
    
    def __len__(self):
        return len(self.valid_index)
    
    def __iter__(self):
        for ind in range(len(self)):
            yield self[ind]
    
    def _read_segments(self, rows2read, output_size):
        output_data = np.zeros((self.embedding_size, output_size), dtype = np.float32)
        with tables.File(self.fname) as fid:
            skels_g = fid.get_node('/embeddings')
            
            tot = 0
            for ini, fin in rows2read:
                dat = skels_g[ini:fin]
                if np.any(np.isnan(dat)):
                    import pdb
                    pdb.set_trace()

                output_data[:, tot:tot+dat.shape[0]] = dat.T
                tot += dat.shape[0] + 1  # the one is because I am leaving all zeros row between chunks
        return output_data

    def _full_video(self, vid_id):
        vid_ranges = self.video_traj_ranges.get_group(vid_id)
        #I might need to do a bit different if I am dealing with larger videos
        
        output_size = vid_ranges['size'].sum() + len(vid_ranges) #the len() part is because I am leaving zeros between chucks
        rows2read = [(row['row_ini'], row['row_fin']) for _, row in vid_ranges.iterrows()]
        return self._read_segments(rows2read, output_size)

    
    def _sample_video(self, vid_id):
        #randomly select trajectories chuncks
        vid_ranges = self.video_traj_ranges.get_group(vid_id)
        tot = 0
        rows2read = []
        while tot < self.sample_size:
            row = vid_ranges.sample(1).iloc[0]
            row_ini = row['row_ini']
            row_fin = row['row_fin']
            size = row['size']
            
            remainder_t = self.sample_size - tot - 1
            top = min(remainder_t, size)
            bot = min(remainder_t, self.min_traj_size)
            size_r = random.randint(bot, top)
            
            ini_r = random.randint(row_ini, row_fin - size_r)
            
            
            rows2read.append((ini_r, ini_r + size_r))
            tot += size_r + 1 # the one is because I am leaving all zeros row between chunks
        
        #read embeddings using the preselected ranges
        return self._read_segments(rows2read, self.sample_size)
        
class SkelTrainer(SkelEmbeddingsFlow):
    def __init__(self, return_label = True, return_snp = False, **argkws):
        super().__init__(**argkws)
        
        data_name = os.path.basename(self.fname)
        if data_name.startswith('CeNDR'):
            self._snps = read_CeNDR_snps()
            valid_strains = self._snps.columns[4:].tolist()
            self._snps[valid_strains] = (self._snps[valid_strains]>0).astype(np.float32)
            self._strain_dict  = {k:ii for ii, k in enumerate(valid_strains)}
            
            self.video_info = self.video_info[self.video_info['strain'].isin(valid_strains)]
        else:
            self._strain_dict = load_strain_dict(self.fname)
        
        
        self.train_index = self.video_info.index[self.video_info['fold'] != self.fold_n_test]
        self.test_index = self.video_info.index[self.video_info['fold'] == self.fold_n_test]

        self.return_label = return_label
        self.return_snp = return_snp
        
        self.train()
    
    @property
    def num_classes(self):
        if self.return_label:
            return len(self._strain_dict)
        elif self.return_snp:
            return self._snps.shape[0]
    
    def __getitem__(self, index):
        vid_id = self.valid_index[index]
        strain = self.video_info.loc[vid_id, 'strain']
        
        out = [self._get_data(vid_id)[None, :, :]]
        
        if  self.return_label:
            out.append(self._strain_dict[strain])
        
        if self.return_snp:
            out.append(self._snps[strain].values)
        
        return out
    
def collate_fn(batch):
    out = [torch.from_numpy(np.stack(x)) for x in zip(*batch)]
    return out


    
 
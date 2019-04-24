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

import pandas as pd
import tables
import numpy as np
import tqdm
import warnings
#%%
def df_to_records(df):
    
    rec = df.to_records(index=False)
    
    #i want to use this to save into pytables, but pytables does not support numpy objects 
    #so i want to cast the type to the longest string size
    new_dtypes = []
    for name, dtype in rec.dtype.descr:
        if dtype.endswith('O'):
            max_size = max(len(str(x)) for x in rec[name])
            new_dtype = '<S{}'.format(max_size)
            new_dtypes.append((name, new_dtype))
        else:
            new_dtypes.append((name, dtype))
    rec = rec.astype(np.dtype(new_dtypes))
    
    return rec
#%%
def _h_angles(skeletons):
    '''
    Get skeletons angles
    '''
    skeletons[np.isnan(skeletons)] = 0
    
    dd = np.diff(skeletons, axis=1)
    angles = np.arctan2(dd[..., 0], dd[..., 1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        angles = np.unwrap(angles, axis=1)

    mean_angles = np.mean(angles, axis=1)
    angles -= mean_angles[:, None]
    
    return angles, mean_angles
#%%
def _h_centred_skels(skeletons, norm_factor = 1000):
    skeletons[np.isnan(skeletons)] = 0
    skel_c = (skeletons - np.mean(skeletons, axis=1)[:, None, :])
    skel_c = np.concatenate((skel_c[..., 0], skel_c[..., ::-1, 1]), axis=skel_c.ndim-2)
    return skel_c/norm_factor
#%%
def _h_eigenworms(skeletons, EIGENWORMS_COMPONENTS, n_components = 6):
    angles, mean_angles = _h_angles(skeletons)
    eigenworms = np.dot(EIGENWORMS_COMPONENTS[:n_components], angles.T)
    return eigenworms.T, mean_angles
    
def _h_eigenworms_full(skeletons, EIGENWORMS_COMPONENTS, n_components = 6):
    '''
    Fully transform the worm skeleton using its eigen components.
    
    The first four vectors are:
        (0,1) the change in head x,y position btw frame
        (2) the change in mean angle btw frames
        (3) each segment mean length
    
    The last six are the eigenvalues    
    
    '''
    
    nan_points = np.isnan(skeletons[:, 0, 0])
    
    eigenworms, mean_angles = _h_eigenworms(skeletons.copy(), EIGENWORMS_COMPONENTS, n_components)
    
    #this method of padding is slow, but i dont want to loose time optimizing it
    mean_angles[nan_points] = np.nan
    delta_ang = np.hstack((0, np.diff(mean_angles)))
    delta_ang[np.isnan(delta_ang)] = 0
    
    #get how much the head position changes over time but first rotate it to the skeletons to 
    #keep the same frame of coordinates as the mean_angles first position
    ang_m = mean_angles[0]
    R = np.array([[np.cos(ang_m), -np.sin(ang_m)], [np.sin(ang_m), np.cos(ang_m)]])
    head_r = skeletons[:, 0, :]
    head_r = np.dot(R, head_r.T)
    
    delta_xy = np.concatenate((np.zeros((2,1)), np.diff(head_r, axis=1)), axis=1).T
    delta_xy[np.isnan(delta_xy)] = 0
    
    #size of each segment (the mean is a bit optional, at this point all the segment should be of equal size)
    L = np.sum(np.linalg.norm(np.diff(skeletons, axis=1), axis=2), axis=1)
    delta_L = np.hstack((0,np.diff(L)))
    delta_L[np.isnan(delta_L)] = 0
    
    #pack all the elments of the transform
    DT = np.concatenate((delta_xy, delta_L[:, None], delta_ang[:, None], eigenworms), axis=1)
    
    return DT    

#%%
def _get_emb_maker(emb_set):
    if emb_set.startswith('eigen'):
        eigen_projection_file = Path(__file__).resolve().parent / 'pca_components.npy'
        assert eigen_projection_file.exists()
        EIGENWORMS_COMPONENTS = np.load(str(eigen_projection_file))
        
    if emb_set  == 'angles':
        n_embeddings = 48
        emb_maker = lambda x : _h_angles(x)[0]
    elif emb_set == 'skeletons':
        n_embeddings = 96
        emb_maker = _h_centred_skels
    elif emb_set == 'eigen':
        n_embeddings = 6
        emb_maker = lambda x : _h_eigenworms(x, EIGENWORMS_COMPONENTS)
    elif emb_set == 'eigenfull':
        n_embeddings = 10  
        emb_maker = lambda x : _h_eigenworms_full(x, EIGENWORMS_COMPONENTS)
    elif emb_set.startswith('AE'):
        #autoencoder return the same values
        n_embeddings = 32
        emb_maker = lambda x : x 
    
    else:
        raise ValueError(emb_set)
    
    return emb_maker, n_embeddings
                            
                            
def calculate_embeddings(video_info, 
                         fnames, 
                         emb_set,
                         save_file,
                         col_label = 'skeleton_id',
                         embeddings_field = '/coordinates/skeletons'
                         ):
    
    emb_maker, n_embeddings = _get_emb_maker(emb_set)
    
    traj_ranges = []
    
    filters = tables.Filters(complevel=0, 
                          complib='blosc', 
                          shuffle=True, 
                          bitshuffle=True, 
                          fletcher32=True
                          )
    
    irow = 0
    with tables.File(save_file, 'w') as fid:
        angle_tab = fid.create_earray('/', 
                                      'embeddings', 
                                      atom = tables.Float32Atom(), 
                                      shape = (0, n_embeddings), 
                                      filters  =filters
                                      )
        
        for ifname, fname in tqdm.tqdm(enumerate(fnames), total=len(fnames)):
            with pd.HDFStore(fname, 'r') as fid:
                trajectories_data = fid['/trajectories_data']
            
            trajectories_data = trajectories_data[['worm_index_joined', 'frame_number', col_label]]
            trajectories_data = trajectories_data[trajectories_data[col_label]>=0]
            
            with tables.File(fname, 'r') as fid:
                emb_g = fid.get_node(embeddings_field)
                for w_ind, dat in trajectories_data.groupby('worm_index_joined'):
                    ini, top = dat[col_label].min(), dat[col_label].max()

                    try:

                        w_emb = emb_g[ini:top]
                        w_emb = emb_maker(w_emb)
                        
                        
                        angle_tab.append(w_emb)
                        angle_tab.append(np.full((1, n_embeddings), np.nan)) #add this to be sure I am not reading between embeddings when processing the data
                        
                        row = (ifname, w_ind,
                               dat['frame_number'].min(), dat['frame_number'].max(),
                               irow, irow + w_emb.shape[0],
                               )
                        irow += w_emb.shape[0] + 1
                        
                        traj_ranges.append(row)
                    except tables.exceptions.HDF5ExtError:
                        print('Error : {}'.format(fname))
                 
    traj_ranges = pd.DataFrame(traj_ranges, columns=['video_id', 'worm_index', 'frame_ini', 'frame_fin', 'row_ini', 'row_fin'])
    
    #%% save data
    video_info_rec = df_to_records(video_info)
    traj_ranges_rec = df_to_records(traj_ranges)
    
    with tables.File(save_file, 'r+') as fid:
        fid.create_table('/', 'trajectories_ranges', obj  =traj_ranges_rec)   
        fid.create_table('/', 'video_info', obj = video_info_rec)
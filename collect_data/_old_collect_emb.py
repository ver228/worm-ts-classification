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

from worm_ts_classification.path import _root_dirs
import pandas as pd
import tables
import numpy as np
import glob
import os
import tqdm
import datetime
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
def get_video_info_from_files(root_dir, f_ext):
    fnames = glob.glob(os.path.join(root_dir, '**', '*' + f_ext), recursive = True)
    fnames = sorted(fnames)
    
    video_info = []
    for fname in fnames:
        bn = os.path.basename(fname).replace(f_ext, '')
        
        parts = bn.split('_')
        parts = parts[:-1] if parts[-1] == 'ROIs' else parts
        
        
        strain = parts[0]
        n_worms = int(parts[1][5:])
        
        dt =datetime.datetime.strptime(parts[-2] + parts[-1], '%d%m%Y%H%M%S')
        date_str = datetime.datetime.strftime(dt, '%Y-%m-%d')
        time_str = datetime.datetime.strftime(dt, '%H:%M:%S')
        
        fname_r = fname.replace(root_dir, '')
        
        set_n = int(fname_r.partition('_Set')[-1][0])
        
        row = (strain, set_n, n_worms, date_str, time_str, fname_r)
        video_info.append(row)
    
    video_info = pd.DataFrame(video_info, columns=['strain', 'set_n', 'n_worms', 'date', 'time', 'file_path'])
    return video_info, fnames

def get_video_info_from_csv_agg(root_dir):
    csv_path = str(Path.home() / 'workspace/WormData/screenings/Serena_WT_Screening/metadata_aggregation_screening.csv')
    bad_labels = ['NONE']
    
    video_info = pd.read_csv(csv_path)
    video_info['dirname'] = video_info['dirname'].str.replace('/Volumes/behavgenom_archive\$/Serena/AggregationScreening/MaskedVideos/', '')
    video_info = video_info.rename(columns={'strain_name':'strain', 'dirname':'file_path'})
    video_info = video_info[~video_info['strain'].isin(bad_labels)]
    

    fnames = root_dir + '/' + video_info['file_path'] + '/'+  video_info['basename'].str.replace('.hdf5', '_featuresN.hdf5')
    is_valid = [os.path.exists(x) for x in fnames.values]

    video_info = video_info[is_valid]
    fnames = [Path(x) for e, x in zip(is_valid,fnames.values) if e]

    return video_info, fnames


if __name__ == '__main__':
    p = 'osx' if sys.platform == 'darwin' else 'centos_oxford'
    root = _root_dirs[p]

    # set_type = 'SWDB'
    # root_dir = str(Path.home() / 'workspace' / 'WormData' / 'screenings')
    # f_ext = '_interpolated25.hdf5'
    # col_label = 'skeleton_id'
    # embeddings_field = '/coordinates/skeletons' 

    
    # set_type = 'CeNDR'
    # root_dir = root + 'screenings/CeNDR/Results'
    # f_ext = '_featuresN.hdf5'
    # col_label = 'skeleton_id'
    # embeddings_field = '/coordinates/skeletons'


#    set_type = 'CeNDRAgg'
#    root_dir = root + 'screenings/Serena_WT_Screening/Results'
#    f_ext = '_featuresN.hdf5'
#    col_label = 'skeleton_id'
#    embeddings_field = '/coordinates/skeletons'


#    set_type = 'pesticides-train'
#    root_dir = root + 'screenings/pesticides/train'
#    f_ext = '_featuresN.hdf5'
#    col_label = 'skeleton_id'
#    embeddings_field = '/coordinates/skeletons'
    
    set_type = 'pesticides-test'
    root_dir = root + 'screenings/pesticides/test'
    f_ext = '_featuresN.hdf5'
    col_label = 'skeleton_id'
    embeddings_field = '/coordinates/skeletons'
    
    
    emb_set = 'angles'
    n_embeddings = 48
    
    # emb_set = 'skeletons'
    # n_embeddings = 49*2
    
    # emb_set = 'eigen'
    # n_embeddings = 6

    # emb_set = 'eigenfull'
    # n_embeddings = 10  

    # set_type = 'CeNDR'
    # emb_set = 'AE2DWithSkels32_emb32_20180620'
    # root_dir = root + 'experiments/autoencoders/embeddings/CeNDR_ROIs_embeddings/20180620_173601_AE2DWithSkels32_skel-1-1_adam_lr0.001_batch16'
    # f_ext = '_embeddings.hdf5'
    # col_label = 'roi_index'
    # n_embeddings = 32
    # embeddings_field = '/embeddings'
    if emb_set.startswith('eigen'):
        eigen_projection_file = Path(__file__).resolve().parent / 'pca_components.npy'
        assert eigen_projection_file.exists()
        EIGENWORMS_COMPONENTS = np.load(str(eigen_projection_file))

    
    #%%
    if set_type == 'CeNDR':
        video_info, fnames = get_video_info_from_files(root_dir, f_ext)
    elif set_type == 'CeNDRAgg':
        video_info, fnames = get_video_info_from_csv_agg(root_dir)
    elif set_type == 'SWDB':
        old_root = '/Volumes/behavgenom_archive\$/'
        csv_file = Path(__file__).resolve().parents[2] / 'single_worm' / 'interpolated_skeletons' / 'strains2process.csv'
        
        video_info = pd.read_csv(str(csv_file))
        video_info['file_path'] = video_info['results_dir'].str.replace(old_root, '') + os.sep + video_info['base_name']
        fnames = [str(Path(root_dir) / (x['file_path'] + f_ext)) for _, x in video_info.iterrows()]
        
        
    #%%
    save_file = root + 'experiments/classify_strains/{}_{}.hdf5'.format(set_type, emb_set)
     
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
                        
                        if emb_set == 'angles':
                            w_emb, _ = _h_angles(w_emb)
                        elif emb_set == 'skeletons':
                            w_emb = _h_centred_skels(w_emb)
                            
                        elif emb_set == 'eigen':
                            w_emb, _ = _h_eigenworms(w_emb, EIGENWORMS_COMPONENTS)
                        elif emb_set == 'eigenfull':
                            w_emb = _h_eigenworms_full(w_emb, EIGENWORMS_COMPONENTS)
                            
                            
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
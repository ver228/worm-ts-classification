#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import itertools
from collections import defaultdict

if __name__ == '__main__':
    root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/papers/eccv2018/figures/output_embeddings/')
    
    embeddings_files = {
            'SWDB' : 'SWDB_angles_embeddings.csv',
            'CeNDR' : 'CeNDR_angles_embeddings.csv'
            }
    
    embeddings = {}
    for set_name, fname in embeddings_files.items():
        emb_df = pd.read_csv(str(root_dir / fname))
        X = emb_df.iloc[:, 2:].values
        X_embedded = TSNE(n_components=2, verbose=1).fit_transform(X)
        df = pd.DataFrame(X_embedded, columns=['X1', 'X2'])
        df['strain'] = emb_df.iloc[:, 1]
        embeddings[set_name] = df
        
    #%%
    
    
    df = embeddings['CeNDR'].copy()
    hue_order = df['strain'].unique()
    gg = itertools.cycle('osv^')
    mm = [v for _,v in zip(hue_order, gg)]
    
    ax = sns.lmplot('X1', 
                    'X2', 
                    hue='strain',
                    data=df, 
                    palette="colorblind", 
                    markers=mm, 
                    fit_reg=False,
                    legend=False, 
                    scatter_kws={"s": 30},
                    size=4
                    )
    ax.add_legend(label_order = hue_order, title='')
    plt.title('MW Dataset')
    plt.savefig('MWtsne.pdf')
    #%%
    df = embeddings['SWDB'].copy()
    
    wt_isolates = ['CB4852','CB4853','CB4856','ED3017','ED3021','ED3049',
                   'ED3054','JU258','JU298','JU343','JU345','JU393','JU394',
                   'JU402','JU438','JU440','LSJ1','MY16','PS312','RC301'
                   ]
    
    unc_strains = ["CB845","HH27","CB1460","CB933","AQ2936","AQ2937","CB262","CB57","CB4870","CB169","CB1069","CB587","VC1528","VC12","CB102","NM1657","FF41","HE130","CB4371","CB566","CB402","CB1068","MP145","CB101","CB1265","VC731","CB109","CB193","DR96","AQ2932","MT1093","CB755","CB1597","CB270","CB5","RB1316","CB81","CB94","AQ2935","AQ2933","CB723","CB189","CB120","AQ2613","DR1089","EG106","DR1","MT324","QT309","CB904","CB950","DR2","AQ2934","MT2611","SP1789","CB15","RW85","CB1416","CB1197","CB1598","CB151","VC854","MT1656","AQ1441"]
    egl_strains = ["MT2068","AQ2316","MT2316","AQ916","KP2018","CE1047","MT1241","MT2246","JR2370","KS99","MT1079","MT6129","CF263","MT1067","MT1543","MT1202","MT2248","MT1216","MT155","MT1083","CB1313","MT1236","MT1200","MT1444","MT1081","MT2293","MT1232","MT1078","MT1217","MT8504","MT2247","MT1179","MT1205","MT1231","MT1082","MT151","MT1222","MT1540"]
    
    
    mm = defaultdict(lambda  : 'Others')
    mm['N2_XX'] = 'N2 Hermaphrodite'
    mm['N2_XO'] = 'N2 Male'
    for wt in wt_isolates:
        mm[wt + '_XX'] = 'Wild Isolates'
    
    for x in unc_strains:
        mm[x + '_XX'] = 'Unc'
        
    for x in egl_strains:
        mm[x + '_XX'] = 'Egl'
    
    s_types = [mm[x] for x in df['strain']]
    
    df['strain'] = s_types
    
    hue_order = ['N2 Male', 'Wild Isolates', 'Unc', 'Egl', 'N2 Hermaphrodite', 'Others']
    ax = sns.lmplot('X1', 'X2', 
               data=df, 
               fit_reg=False, 
               hue='strain', 
               hue_order=hue_order[::-1], 
               legend_out=True, 
               palette="Paired", 
               scatter_kws={"s": 20},
               legend=False,
               size=4)
    
    ax.add_legend(label_order =[ 'N2 Hermaphrodite', 'N2 Male', 'Wild Isolates', 'Unc', 'Egl',  'Others'], title='')
    plt.title('SW Dataset')
    plt.savefig('SWtsne.pdf')
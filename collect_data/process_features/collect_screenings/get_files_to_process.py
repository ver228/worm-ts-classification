#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:17:36 2018

@author: avelinojaver
"""
from pathlib import Path
import subprocess
import tqdm

SPLIT_TOKEN = '\n*********************************************\n'
SRC_FILE = Path('./screens_list.txt')
DST_FILE = Path.home() / 'workspace/tierpsy2process.txt'

CMD_PROCESS = '''#!/bin/bash

#$ -P rittscher.prjc -q short.qc
#$ -t 1-{} -pe shmem 2



module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0
source activate tierpsy

FILESSOURCE="{}"

echo "Username: " `whoami`
FSOURCE=$(awk "NR==$SGE_TASK_ID" $FILESSOURCE)
FSOURCE="OMP_NUM_THREADS=2 $(awk "NR==$SGE_TASK_ID" $FILESSOURCE)"
echo $FSOURCE
eval "$FSOURCE"

exit 0
'''

CMD2SAVE = Path('./process_all.sh')
#%%

if __name__ == '__main__':
    
    all_cmds = []
    with SRC_FILE.open() as fid:
        dat = [x.split() for x in fid.read().split('\n') if x]
    
    #%%
    for screen_name, params_file, mask_dir in tqdm.tqdm(dat):
        mask_dir = str(Path(mask_dir).expanduser())
        
        cmd = f'source activate tierpsy; tierpsy_process --mask_dir_root "{mask_dir}"  --tmp_dir_root "" --only_summary --json_file "{params_file}" --is_debug --pattern_exclude "*_interpolated25.hdf5"'
        
        if params_file.startswith('WT2'):
            cmd += f' --results_dir_root "{mask_dir}"'

        result = subprocess.check_output(cmd, shell=True)
        result = result.decode('utf-8')
        
        parts = result.split(SPLIT_TOKEN)
        dd =  [x.partition(' ')[-1] for x in parts[4].split('\n')]
        all_cmds += ['python ' +  x for x in dd if x]
        
    #%%
    
    with DST_FILE.open('w') as fid:
        fid.write('\n'.join(all_cmds))
    #%%
    
    with CMD2SAVE.open('w') as fid:
        out = CMD_PROCESS.format(len(all_cmds), str(DST_FILE.resolve()))
        fid.write(out)
    
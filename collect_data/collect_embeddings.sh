#!/bin/bash

#$ -P rittscher.prjc -q short.qc
#module use -a /mgmt/modules/eb/modules/all
#module load Anaconda3/5.1.0

echo "Username: " `whoami`
source activate pytorch-0.4.1
python /users/rittscher/avelino/GitLab/worm-ts-classification/collect_data/collect_emb_pesticides.py
exit 0

#!/bin/bash

#$ -P rittscher.prjc -q short.qc -pe shmem 4


echo "Username: " `whoami`
source $HOME/ini_session.sh
source activate tierpsy
python /users/rittscher/avelino/GitLab/worm-ts-classification/collect_data/features/collect_features.py
exit 0

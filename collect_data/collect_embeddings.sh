#!/bin/bash

#$ -P rittscher.prjc -q short.qc


echo "Username: " `whoami`
source $HOME/ini_session.sh
python /users/rittscher/avelino/GitLab/worm-ts-classification/collect_data/collect_emb.py
exit 0

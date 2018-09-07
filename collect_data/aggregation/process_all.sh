#!/bin/bash

#$ -P rittscher.prjc -q short.qc
#$ -t 1-530 -pe shmem 2

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0
source activate tierpsy

#FILESSOURCE="$HOME/worm-ts-classification/collect_data/aggregation/files2process.txt"
FILESSOURCE="$HOME/workspace/files2process.txt"

echo "Username: " `whoami`
FSOURCE=$(awk "NR==$SGE_TASK_ID" $FILESSOURCE)
echo $FSOURCE
eval $FSOURCE
exit 0
#!/bin/bash

#$ -P rittscher.prjc -q short.qc
#$ -t 1-114 -pe shmem 2

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0
source activate tierpsy

FILESSOURCE="/Volumes/rescomp1/data/files2process.txt"

echo "Username: " `whoami`
FSOURCE=$(awk "NR==$SGE_TASK_ID" $FILESSOURCE)
echo $FSOURCE

exit 0

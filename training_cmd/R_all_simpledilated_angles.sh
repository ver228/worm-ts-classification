#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q@compG004 -l gputype=p100
#$ -l gpu=1 -pe shmem 1

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0
source activate pytorch-v0.4.0-cuda8.0-venv 

echo "Username: " `whoami`
echo $HOME
echo cuda_id: $CUDA_VISIBLE_DEVICES

SCRIPTPATH="$HOME/GitLab/worm-ts-classification/scripts/train_model.py"
python $SCRIPTPATH \
--model_name 'simpledilated' --set_type 'angles' --n_epochs 1000 --batch_size 8 \
--num_workers 1 --optimizer 'sgd' --lr 0.0001 \
--copy_tmp '/tmp/avelino'$CUDA_VISIBLE_DEVICES \
--init_model_path 'logs/angles_20180530_010133_simpledilated_sgd_lr0.001_batch8/model_best.pth.tar'


echo "Finished at :"`date`
exit 0


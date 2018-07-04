#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -l gputype=p100
#$ -l gpu=1 -pe shmem 1

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0
source activate pytorch-v0.4.0-cuda8.0-venv 

echo "Username: " `whoami`
echo $HOME
echo cuda_id: $CUDA_VISIBLE_DEVICES


#AE_emb_20180206
SCRIPTPATH="$HOME/GitLab/worm-ts-classification/worm_ts_classification/trainer.py"
python $SCRIPTPATH --is_divergent_set \
--model_name 'simpledilated' --set_type 'AE2DWithSkels32_emb32_20180620' --n_epochs 1000 --batch_size 8 \
--num_workers 1 --optimizer 'adam' --lr 0.00001 --copy_tmp '/tmp/avelino'$CUDA_VISIBLE_DEVICES 

echo "Finished at :"`date`

exit 0

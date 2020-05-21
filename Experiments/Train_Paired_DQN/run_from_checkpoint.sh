#!/bin/sh

BASE_DIR="../../Logs/Rainbow/Paired/"
TARGET_FOLDER="20200515-114852Mirror/checkpoints/"
CUR_DATE=`date "+%Y%m%d-%H%M%S"`
LOG_PATH="$BASE_DIR$TARGET_FOLDER"
NEW_LOG_PATH="$BASE_DIR$TARGET_FOLDER/$CUR_DATE"

TRAINING_PARTNERS="all"
EVAL_PARTNERS="all"




export PYTHONPATH=${PYTHONPATH}:../..
export PYTHONPATH=${PYTHONPATH}:../../agents/rainbow
export PYTHONPATH=${PYTHONPATH}:../../Experiments/Rulebased
# export CUDA_VISIBLE_DEVICES=""



python3 -um train_paired \
  --base_dir=${LOG_PATH} \
  --gin_files="hanabi_rainbow.gin"\
  --checkpoint_dir=${LOG_PATH} \
  --checkpoint_version=1000 \
  --training_partners=${TRAINING_PARTNERS} \
  --eval_partners=${EVAL_PARTNERS} \
  --lenient="False"

  #--gin_bindings='RainbowAgent'

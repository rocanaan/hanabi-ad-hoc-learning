#!/bin/sh

BASE_DIR="../../Logs/Rainbow/"
CUR_DATE=`date "+%Y%m%d-%H%M%S"`
LOG_PATH="$BASE_DIR$CUR_DATE"
# CHECKPOINT_DIR="${BASE_DIR}20190906-153116/checkpoints"


export PYTHONPATH=${PYTHONPATH}:../..
export PYTHONPATH=${PYTHONPATH}:../../agents/rainbow
export PYTHONPATH=${PYTHONPATH}:../../Experiments/Rulebased


python3 -um evaluate_paired \
  --base_dir=${LOG_PATH} \
  --gin_files="hanabi_rainbow.gin"
  --gin_bindings='RainbowAgent' 
  --checkpoint_dir=${CHECKPOINT_DIR} 

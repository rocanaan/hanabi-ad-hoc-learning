#!/bin/sh

BASE_DIR="../../Logs/Rainbow/"
CUR_DATE=`date "+%Y%m%d-%H%M%S"`
LOG_PATH="$BASE_DIR$CUR_DATE"
CHECKPOINT_DIR="${BASE_DIR}20190906-153116/checkpoints"


export PYTHONPATH=${PYTHONPATH}:../..
export PYTHONPATH=${PYTHONPATH}:../../agents/rainbow
mkdir -p ${CHECKPOINT_DIR}


python3 -um train \
  --base_dir=${LOG_PATH} \
  --gin_files="../../agents/rainbow/configs/hanabi_rainbow.gin" \
  --checkpoint_dir=${CHECKPOINT_DIR} \

  #--gin_bindings='RainbowAgent' \
  
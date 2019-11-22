#!/bin/sh

BASE_DIR="../../Logs/Rainbow/"
CUR_DATE=`date "+%Y%m%d-%H%M%S"`
LOG_PATH="$BASE_DIR$CUR_DATE"

export PYTHONPATH=${PYTHONPATH}:../..

python -um train \
  --base_dir=${LOG_PATH}
  --gin_files='configs/hanabi_rainbow.gin'
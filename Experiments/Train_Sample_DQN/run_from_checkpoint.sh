#!/bin/sh

BASE_DIR="../../Logs/Rainbow/"
TARGET_FOLDER="20191110-233342"
CUR_DATE=`date "+%Y%m%d-%H%M%S"`
LOG_PATH="$BASE_DIR$TARGET_FOLDER"
NEW_LOG_PATH="$BASE_DIR$TARGET_FOLDER/$CUR_DATE"




export PYTHONPATH=${PYTHONPATH}:../..
export PYTHONPATH=${PYTHONPATH}:../../agents/rainbow

python3 -um train \
  --base_dir=${LOG_PATH} \
  --gin_files="../../agents/rainbow/configs/hanabi_rainbow.gin"\
  --checkpoint_dir=${LOG_PATH}} \
  --checkpoint_save_dir=${NEW_LOG_PATH}\
  --checkpoint_version=3300
  --gin_bindings='RainbowAgent'

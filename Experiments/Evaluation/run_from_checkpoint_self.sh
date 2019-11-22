#!/bin/sh

BASE_DIR="../../Logs/Rainbow/"
TARGET_FOLDER="20191031-133819/20191031-141227"
CUR_DATE=`date "+%Y%m%d-%H%M%S"`
LOG_PATH="$BASE_DIR$TARGET_FOLDER"
NEW_LOG_PATH="$BASE_DIR$TARGET_FOLDER/$CUR_DATE"




export PYTHONPATH=${PYTHONPATH}:../..
export PYTHONPATH=${PYTHONPATH}:../../agents/rainbow

python3 -um evaluate \
  --base_dir=:"Outer3950" \
  --gin_files="hanabi_rainbow_self.gin" \
  --checkpoint_dir="Paired/Outer3950" \
  --num_of_iterations=20 \
  #--checkpoint_version=100
  --gin_bindings='RainbowAgent'\
  --checkpoint_save_dir=${NEW_LOG_PATH}

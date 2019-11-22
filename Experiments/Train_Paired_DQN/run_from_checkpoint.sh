#!/bin/sh

BASE_DIR="../../Logs/Rainbow/Paired/"
TARGET_FOLDER="Ensemble5500"
CUR_DATE=`date "+%Y%m%d-%H%M%S"`
LOG_PATH="$BASE_DIR$TARGET_FOLDER"
NEW_LOG_PATH="$BASE_DIR$TARGET_FOLDER/$CUR_DATE"




export PYTHONPATH=${PYTHONPATH}:../..
export PYTHONPATH=${PYTHONPATH}:../../agents/rainbow
export PYTHONPATH=${PYTHONPATH}:../../Experiments/Rulebased
# export CUDA_VISIBLE_DEVICES=""



python3 -um train_paired \
  --base_dir=${LOG_PATH} \
  --gin_files="hanabi_rainbow.gin"\
  --checkpoint_dir=${LOG_PATH} \
  --checkpoint_version=5500
  #--checkpoint_dir="/home/jupyter/Notebooks/Rodrigo/PairedToTrain/VDB_5050_1283"

  #--gin_bindings='RainbowAgent'

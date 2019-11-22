#!/bin/sh

BASE_DIR="../../Logs/Rainbow/Paired/"
CUR_DATE=`date "+%Y%m%d-%H%M%S"`
LOG_PATH="$BASE_DIR$CUR_DATE"
# CHECKPOINT_DIR="${BASE_DIR}20190906-153116/checkpoints"


export PYTHONPATH=${PYTHONPATH}:../..
export PYTHONPATH=${PYTHONPATH}:../../agents/rainbow
export PYTHONPATH=${PYTHONPATH}:../../Experiments/Rulebased
export PYTHONPATH=${PYTHONPATH}:/home/jupyter/Notebooks/Rodrigo/hanabilearningenvironment/
export PYTHONPATH=${PYTHONPATH}:/home/jupyter/Notebooks/Rodrigo/hanabilearningenvironment/Experiments/Rulebased
export PYTHONPATH=${PYTHONPATH}:/home/jupyter/Notebooks/Rodrigo/hanabilearningenvironment/agents/rainbow
# export CUDA_VISIBLE_DEVICES=""



echo $PYTHONPATH$

python -um train_paired \
  --base_dir=${LOG_PATH} \
  --gin_files="hanabi_rainbow.gin"
  --gin_bindings='RainbowAgent' 
  --checkpoint_dir=${CHECKPOINT_DIR} 
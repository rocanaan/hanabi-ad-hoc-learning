#!/bin/sh

BASE_DIR="../../Logs/Rainbow/Paired/"
CUR_DATE=`date "+%Y%m%d-%H%M%S"`
LOG_PATH="$BASE_DIR$CUR_DATE"
CHECKPOINT_DIR="${LOG_PATH}/checkpoints"

# Training and evalu partners can be either "rb" all"
TRAINING_PARTNERS="Mirror"
EVAL_PARTNERS="all"


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
  --gin_files="hanabi_rainbow.gin"\
  --checkpoint_dir=${CHECKPOINT_DIR} \
  --training_partners=${TRAINING_PARTNERS} \
  --eval_partners=${EVAL_PARTNERS} \
  --lenient="True"
  # --gin_bindings='RainbowAgent' \

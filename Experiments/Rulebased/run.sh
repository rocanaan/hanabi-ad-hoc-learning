#!/bin/sh

BASE_DIR="../../Logs/Rainbow/"
CUR_DATE=`date "+%Y%m%d-%H%M%S"`
LOG_PATH="$BASE_DIR$CUR_DATE"
CHECKPOINT_DIR="${BASE_DIR}20190906-153116/checkpoints"


export PYTHONPATH=${PYTHONPATH}:../..
export PYTHONPATH=${PYTHONPATH}:../../agents/rainbow

python -um rl_env_example --num_episodes 1000 --agent_class VanDenBerghAgent
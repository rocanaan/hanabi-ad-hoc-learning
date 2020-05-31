#!/bin/sh

# Training and evalu partners can be either "rb" all"
# TRAINING_PARTNERS="all"
# TRAINING_PARTNERS="rb"
TRAINING_PARTNERS="Mirror"
# TRAINING_PARTNERS="InternalAgent"
# TRAINING_PARTNERS="OuterAgent"
# TRAINING_PARTNERS="IGGIAgent"
# TRAINING_PARTNERS="LegalRandomAgent"
# TRAINING_PARTNERS="FlawedAgent"
# TRAINING_PARTNERS="PiersAgent"
# TRAINING_PARTNERS="VanDenBerghAgent"


#Internal variations
# TRAINING_PARTNERS="InternalDiscardOldest"
# TRAINING_PARTNERS="InternalProbabilistic"
# TRAINING_PARTNERS="InternalSwapped"



# For  training will all agents except one
# TRAINING_PARTNERS="InternalAgent_OuterAgent_IGGIAgent_FlawedAgent_PiersAgent_VanDenBerghAgent_Mirror" # All 
# TRAINING_PARTNERS="OuterAgent_IGGIAgent_FlawedAgent_PiersAgent_VanDenBerghAgent_Mirror" # No Internal
# TRAINING_PARTNERS="InternalAgent_IGGIAgent_FlawedAgent_PiersAgent_VanDenBerghAgent_Mirror" # No Outer 
# TRAINING_PARTNERS="InternalAgent_OuterAgent_FlawedAgent_PiersAgent_VanDenBerghAgent_Mirror" # No IGGI 
# TRAINING_PARTNERS="InternalAgent_OuterAgent_IGGIAgent_PiersAgent_VanDenBerghAgent_Mirror" # No Flawed 
# TRAINING_PARTNERS="InternalAgent_OuterAgent_IGGIAgent_FlawedAgent_VanDenBerghAgent_Mirror" # No Piers 
# TRAINING_PARTNERS="InternalAgent_OuterAgent_IGGIAgent_FlawedAgent_PiersAgent_Mirror" # No VDB 
# TRAINING_PARTNERS="InternalAgent_OuterAgent_IGGIAgent_FlawedAgent_PiersAgent_VanDenBerghAgent" # No mirror, same as rb 

# Some possible sets of training partners that share a certain feature
# TRAINING_PARTNERS="FlawedAgent_PiersAgent_VanDenBerghAgent" # Only probabilistic plays
# TRAINING_PARTNERS="InternalAgent_OuterAgent_IGGIAgent" # Only predictable plays 




EVAL_PARTNERS="all"

BASE_DIR="../../Logs/Rainbow/Paired/"
CUR_DATE=`date "+%Y%m%d-%H%M%S"`
LOG_PATH="$BASE_DIR$CUR_DATE$TRAINING_PARTNERS"
CHECKPOINT_DIR="${LOG_PATH}/checkpoints"


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
  --lenient="False"
  # --gin_bindings='RainbowAgent' \

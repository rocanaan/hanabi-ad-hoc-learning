#!/bin/sh

# BASE_DIR="../../Logs/Rainbow/"
# TARGET_FOLDER="20191024-150717"
# CUR_DATE=`date "+%Y%m%d-%H%M%S"`
# LOG_PATH="$BASE_DIR$TARGET_FOLDER"
# NEW_LOG_PATH="$BASE_DIR$TARGET_FOLDER/$CUR_DATE"




export PYTHONPATH=${PYTHONPATH}:../..
export PYTHONPATH=${PYTHONPATH}:../../agents/rainbow
export PYTHONPATH=${PYTHONPATH}:../Rulebased
export PYTHONPATH=${PYTHONPATH}:../Train_Paired_DQN

python3 -um behavioral_evaluation \
  --my_agents=AIIDE_agents.txt \
  --their_agents=rulebased_agents.txt \
  --num_of_iterations=1000   \
  --gin_files="hanabi_rainbow.gin"\

  # --base_dir=${LOG_PATH} \
  # --gin_files="hanabi_rainbow.gin" \
  # --checkpoint_dir="Paired/Outer5500" \
  # --num_of_iterations=10
#--train_only="IGGIAgent"
 #--evaluate_all=1 \
 #--agent1="RainbowAgent" \
 #--agent2="RainbowAgent" \
 #  --self_gin_files = "hanabi_rainbow_self.gin" \
  
  #--checkpoint_version=0 \

  #--checkpoint_save_dir=${NEW_LOG_PATH}\
  #--checkpoint_version=100
  #--gin_bindings="RainbowAgent"

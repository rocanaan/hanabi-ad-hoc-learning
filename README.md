Run [run_from_checkpoint_rule_based.sh](Experiments/Evaluation/run_from_checkpoint_rule_based.sh) to get performance of rainbow agents vs. rule-based agents.
Run [run_from_checkpoint_rainbow.sh](Experiments/Evaluation/run_from_checkpoint_rainbow.sh) to get performance of rainbow agents vs. rainbow agents.

This work is based on rainbow agent described below. 

This is not an officially supported Google product.

hanabi\_learning\_environment is a research platform for Hanabi experiments. The file rl\_env.py provides an RL environment using an API similar to OpenAI Gym. A lower level game interface is provided in pyhanabi.py for non-RL methods like Monte Carlo tree search.

### Getting started
```
sudo apt-get install g++         # if you don't already have a CXX compiler
sudo apt-get install cmake       # if you don't already have CMake
sudo apt-get install python-pip  # if you don't already have pip
pip install cffi                 # if you don't already have cffi
cmake .
make
python rl_env_example.py         # Runs RL episodes
python game_example.py           # Plays a game using the lower level interface
```

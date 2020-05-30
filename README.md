This is a project based on DeepMind's [Hanabi Learning Environment](https://github.com/deepmind/hanabi-learning-environment) with a focus on learning for ad-hoc teamplay). The main additions are:


### Repository structure
**/Experiments/Rulebased** contains implementation of some Hanabi Rule-based agents previously [introduced](https://ieeexplore.ieee.org/abstract/document/7969465?casa_token=nZ6Xo2g4Oa8AAAAA:dH_qOTpj0oW3e3dw5PI5JitfZANyiv2N0SCL-0Th0PuMgQbpKeMQw9CL18mcjf10hbDYvUs1pNY) by Walton-Rivers et al. and used in the [Hanabi CoG Competition](https://ieeexplore.ieee.org/abstract/document/8848008?casa_token=FuxNAoKlnHIAAAAA:TDxVLxjnanh5dzfiB-BLZzxhVWSgc62RF5C-cKhs4L24nKdjMzvQ1uTzosCHI9-VdmXfu2Yc_v8).
**/Experiments/Train_Paired_DQN**  integrates rule-based agents in the Rainbow training procedure to train variants of Rainbow specialized at playing with any subset of these agents.
**/Experiments/Behavioral_Evaluation** performs behavioral analysis of the games played, keeping track of metrics such as communicativeness, information per play (IPP) and number of correct/incorrect plays. Communicativeness and IPP were first introduced in this [related paper](https://arxiv.org/abs/2004.13710)





Run [run_from_checkpoint_rule_based.sh](Experiments/Evaluation/run_from_checkpoint_rule_based.sh) to get performance of rainbow agents vs. rule-based agents.
Run [run_from_checkpoint_rainbow.sh](Experiments/Evaluation/run_from_checkpoint_rainbow.sh) to get performance of rainbow agents vs. rainbow agents.

### Original readme by DeepMind follows:

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

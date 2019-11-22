# coding=utf-8
# Copyright 2018 The Dopamine Authors and Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
# This file is a fork of the original Dopamine code incorporating changes for
# the multiplayer setting and the Hanabi Learning Environment.
#
"""The entry point for running a Rainbow agent on Hanabi."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import statistics

import rl_env
from third_party.dopamine import logger
from internal_agent import InternalAgent
from agents.random_agent import RandomAgent
from agents.simple_agent import SimpleAgent
from internal_agent import InternalAgent
from rulebased_agent import RulebasedAgent
import run_paired_experiment

AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent, 'InternalAgent': InternalAgent,'RulebasedAgent':RulebasedAgent, 'RainbowAgent':None}
SETTINGS = {'players': 2, 'num_episodes': 100, 'agent_class1': 'SimpleAgent', 'agent_class2': 'RandomAgent'}

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    'gin_files', [],
    'List of paths to gin configuration files (e.g.'
    '"configs/hanabi_rainbow.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1").')

flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')

flags.DEFINE_string('checkpoint_dir', '',
                    'Directory where checkpoint files should be saved. If '
                    'empty, no checkpoints will be saved.')
flags.DEFINE_string('checkpoint_file_prefix', 'ckpt',
                    'Prefix to use for the checkpoint files.')
flags.DEFINE_string('logging_dir', '',
                    'Directory where experiment data will be saved. If empty '
                    'no checkpoints will be saved.')
flags.DEFINE_string('logging_file_prefix', 'log',
                    'Prefix to use for the log files.')
flags.DEFINE_string('checkpoint_save_dir',None,
                    'Path to save directory')
flags.DEFINE_string('checkpoint_version', None,
                    'Specific checkpoint file version to be loaded. If empty, the newest checkpoint will be loaded.')
flags.DEFINE_string('agent1',None,'name of agent1')
flags.DEFINE_string('agent2','InternalAgent','name of agent2')

def launch_experiment(agentX):
  """Launches the experiment.

  Specifically:
  - Load the gin configs and bindings.
  - Initialize the Logger object.
  - Initialize the environment.
  - Initialize the observation stacker.
  - Initialize the agent.
  - Reload from the latest checkpoint, if available, and initialize the
    Checkpointer object.
  - Run the experiment.
  """
  if FLAGS.base_dir == None:
    raise ValueError('--base_dir is None: please provide a path for '
                     'logs and checkpoints.')

  run_paired_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)


  environment = run_paired_experiment.create_environment()
  obs_stacker = run_paired_experiment.create_obs_stacker(environment)
  my_agent = run_paired_experiment.create_agent(environment, obs_stacker,'Rainbow')
  if agentX!='RainbowAgent':
    their_agent = AGENT_CLASSES[agentX]({})
  else:
    their_agent = my_agent

  checkpoint_dir = '{}/checkpoints'.format(FLAGS.base_dir)
  if FLAGS.checkpoint_save_dir == None:
    checkpoint_save_dir = checkpoint_dir
    experiment_logger = logger.Logger('{}/logs'.format(FLAGS.base_dir))
  else:
    checkpoint_save_dir = '{}/checkpoints'.format(FLAGS.checkpoint_save_dir)
    experiment_logger = logger.Logger('{}/logs'.format(FLAGS.checkpoint_save_dir))
    print ("set save dir as: "+ checkpoint_save_dir)
    
  if agentX == 'RainbowAgent':
    start_iteration, experiment_checkpointer = (
        run_paired_experiment.initialize_checkpointing(their_agent,
                                                experiment_logger,
                                                checkpoint_dir,
                                                checkpoint_save_dir,
                                                FLAGS.checkpoint_version,
                                                FLAGS.checkpoint_file_prefix))
                                                
  start_iteration, experiment_checkpointer = (
      run_paired_experiment.initialize_checkpointing(my_agent,
                                              experiment_logger,
                                              checkpoint_dir,
                                              checkpoint_save_dir,
                                              FLAGS.checkpoint_version,
                                              FLAGS.checkpoint_file_prefix))



  return_collection = run_paired_experiment.run_paired_experiment(my_agent, their_agent, environment, start_iteration,
                                obs_stacker,
                                experiment_logger, experiment_checkpointer,
                                checkpoint_save_dir,
                                logging_file_prefix=FLAGS.logging_file_prefix)
  return return_collection

class Runner(object):
  """Runner class."""

  def __init__(self, flags1):
    """Initialize runner."""
    self.flags1 = flags1
    self.agent_config = {'players': flags1['players']}
    self.environment = rl_env.make('Hanabi-Full', num_players=flags1['players'])
    self.agent_class1 = AGENT_CLASSES[flags1['agent_class1']]
    self.agent_class2 = AGENT_CLASSES[flags1['agent_class2']]

  def run(self):
    """Run episodes."""
    rewards = []
    for episode in range(flags1['num_episodes']):
      observations = self.environment.reset()
      agents = [self.agent_class1(self.agent_config), self.agent_class2(self.agent_config)]
      done = False
      episode_reward = 0
      while not done:
        for agent_id, agent in enumerate(agents):
          observation = observations['player_observations'][agent_id]
          try:
            action = agent.act(observation)
          except:
            action = agent.get_move(observation)
          if observation['current_player'] == agent_id:
            assert action is not None
            current_player_action = action
          else:
            assert action is None
        # Make an environment step.
        #print('Agent: {} action: {}'.format(observation['current_player'],
        #                                    current_player_action))
        observations, reward, done, unused_info = self.environment.step(
            current_player_action)
        if (reward >=0):
          episode_reward += reward
      rewards.append(episode_reward)
      print('Running episode: %d' % episode)
      print('Max Reward: %.3f' % max(rewards))
      print('Average Reward: %.3f' % (sum(rewards)/(episode+1)))
    return rewards

  


def main(unused_argv):
  """This main function acts as a wrapper around a gin-configurable experiment.

  Args:
    unused_argv: Arguments (unused).
  """
  for (agent_1 in AGENT_CLASSES):
    for (agent_2 in AGENT_CLASSES):
      return_collection = [0,0]
      if (agent_1 == 'RainbowAgent'):
        return_collection = launch_experiment(agent_2)
      elif(agent_2 == 'RainbowAgent'):
        return_collection = launch_experiment(agent_1)
      else:
        flags1['agent_class1'] = agent_1
        flags1['agent_class2'] = agent_2
        options = [(k, v) for k, v in flags1.items()]
        runner = Runner(flags1)
        return_collection = runner.run()
      print("Average score and std: ")
      print(statistics.mean(return_collection),statistics.stdev(return_collection))
if __name__ == '__main__':
  flags1 = SETTINGS
  options = [(k, v) for k, v in flags1.items()]

  for flag, value in options[1:]:
    #flag = flag[2:]  # Strip leading --.
    flags1[flag] = type(flags1[flag])(value)

  app.run(main)
  

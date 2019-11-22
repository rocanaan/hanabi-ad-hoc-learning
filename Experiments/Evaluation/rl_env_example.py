# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple episode runner using the RL environment."""

from __future__ import print_function

import sys
import getopt
import rl_env
from agents.random_agent import RandomAgent
from agents.simple_agent import SimpleAgent
from internal_agent import InternalAgent
from rulebased_agent import RulebasedAgent

AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent, 'InternalAgent': InternalAgent,'RulebasedAgent':RulebasedAgent}

SETTINGS = {'players': 2, 'num_episodes': 10, 'agent_class1': 'SimpleAgent', 'agent_class2': 'RandomAgent'}

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
        print('Agent: {} action: {}'.format(observation['current_player'],
                                            current_player_action))
        observations, reward, done, unused_info = self.environment.step(
            current_player_action)
        if (reward >=0):
          episode_reward += reward
      rewards.append(episode_reward)
      print('Running episode: %d' % episode)
      print('Max Reward: %.3f' % max(rewards))
      print('Average Reward: %.3f' % (sum(rewards)/(episode+1)))
    return rewards

if __name__ == "__main__":
  flags1 = SETTINGS
  options = [(k, v) for k, v in SETTINGS.items()]
  print(options)

  for flag, value in options[1:]:
    #flag = flag[2:]  # Strip leading --.
    flags1[flag] = type(flags1[flag])(value)

  
  runner = Runner(SETTINGS)
  runner.run()
 
 
 #[('--num_episodes', '100'), ('--agent_class1', 'InternalAgent'), ('--agent_class2', 'RandomAgent')]

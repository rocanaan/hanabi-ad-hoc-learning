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
import numpy as np
from rulebased_agent import RulebasedAgent
from agents.random_agent import RandomAgent
from agents.simple_agent import SimpleAgent
from internal_agent import InternalAgent
from outer_agent import OuterAgent
from iggi_agent import IGGIAgent
from legal_random_agent import LegalRandomAgent
from flawed_agent import FlawedAgent
from piers_agent import PiersAgent
from van_den_bergh_agent import VanDenBerghAgent

# AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent, 'InternalAgent': InternalAgent, 
# 'OuterAgent': OuterAgent,'IGGIAgent':IGGIAgent,'LegalRandomAgent':LegalRandomAgent,'FlawedAgent':FlawedAgent,
# 'PiersAgent':PiersAgent, 'VanDenBerghAgent':VanDenBerghAgent}

AGENT_CLASSES = {'InternalAgent': InternalAgent, 
'OuterAgent': OuterAgent,'IGGIAgent':IGGIAgent,'FlawedAgent':FlawedAgent,
'PiersAgent':PiersAgent, 'VanDenBerghAgent':VanDenBerghAgent}


LENIENT = False

class Runner(object):
  """Runner class."""

  def __init__(self, flags, a1, a2):
    """Initialize runner."""
    self.flags = flags
    self.agent_config = {'players': flags['players']}
    self.environment = rl_env.make('Hanabi-Full', num_players=flags['players'])
    self.agent_class = AGENT_CLASSES[flags['agent_class']]
    self.class1 = AGENT_CLASSES[a1]
    self.class2 = AGENT_CLASSES[a2]

  # def ffa(self, index1, index2):
  #   """Run episodes."""
  #   rewards = []
  #   agents = [self.agent_class(self.agent_config)
  #               for _ in range(self.flags['players'])]
  #   for episode in range(flags['num_episodes']):
  #     observations = self.environment.reset()
  #     done = False
  #     episode_reward = 0
  #     while not done:
  #       for agent_id, agent in enumerate(agents):
  #         observation = observations['player_observations'][agent_id]
  #         action = agent.act(observation)
  #         if observation['current_player'] == agent_id:
  #           assert action is not None
  #           current_player_action = action
  #         else:
  #           assert action is None
  #       # Make an environment step.
  #       # print('Agent: {} action: {}'.format(observation['current_player'],
  #                                           current_player_action))
  #       observations, reward, done, unused_info = self.environment.step(
  #           current_player_action)
  #       if (reward >=0):
  #         episode_reward += reward
  #     rewards.append(episode_reward)
  #     print('Running episode: %d' % episode)
  #     print('Reward of this episode: %d' % episode_reward)
  #     print('Max Reward: %.3f' % max(rewards))
  #     print('Average Reward: %.3f' % (sum(rewards)/(episode+1)))
  #   for a in agents:
  #     a.rulebased.print_histogram()
  #   return rewards

  def run(self):
    """Run episodes."""
    rewards = []
    # agents = [self.agent_class(self.agent_config)
    #             for _ in range(self.flags['players'])]
    # agents = [a1,a2]
    for episode in range(flags['num_episodes']):
      if np.random.uniform() <= 0.5:
        agents = [self.class1(self.agent_config),self.class2(self.agent_config)]
      else:
        agents = [self.class2(self.agent_config),self.class1(self.agent_config)]
      observations = self.environment.reset()
      done = False
      episode_reward = 0
      first_turn = True
      while not done:
        for agent_id, agent in enumerate(agents):
          observation = observations['player_observations'][agent_id]
          if first_turn == True:
            # print(first_turn)
            # print(observation['current_player'])
            first_turn = False
          action = agent.act(observation)
          if observation['current_player'] == agent_id:
            assert action is not None
            current_player_action = action
          else:
            assert action is None
        # Make an environment step.
        # # print('Agent: {} action: {}'.format(observation['current_player'],
        #                                     current_player_action))
        observations, reward, done, unused_info = self.environment.step(
            current_player_action)
        if (reward >=0 or not LENIENT):
          episode_reward += reward

      rewards.append(episode_reward)
      # print('Running episode: %d' % episode)
      # print('Reward of this episode: %d' % episode_reward)
      # print('Max Reward: %.3f' % max(rewards))
      # print('Average Reward: %.3f' % (sum(rewards)/(episode+1)))
    for a in agents:
      a.rulebased.print_histogram()
    return rewards

if __name__ == "__main__":
  flags = {'players': 2, 'num_episodes': 1, 'agent_class': 'SimpleAgent'}
  options, arguments = getopt.getopt(sys.argv[1:], '',
                                     ['players=',
                                      'num_episodes=',
                                      'agent_class='])
  if arguments:
    sys.exit('usage: rl_env_example.py [options]\n'
             '--players       number of players in the game.\n'
             '--num_episodes  number of game episodes to run.\n'
             '--agent_class   {}'.format(' or '.join(AGENT_CLASSES.keys())))
  for flag, value in options:
    flag = flag[2:]  # Strip leading --.
    flags[flag] = type(flags[flag])(value)
  results = []
  for name1 in AGENT_CLASSES:
    for name2 in AGENT_CLASSES:
      runner = Runner(flags, name1, name2)
      reward = np.average(runner.run())
      results.append([name1, name2, reward])

  for r in results:
    print(r)
  print(len(results)) 
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


import run_paired_experiment

from rainbow_agent import RainbowAgent

class BehavioralRunner(object):
  """Runner class."""

  def __init__(self, flags, a1, a2, lenient=False):
    """Initialize runner."""
    self.flags = flags
    self.agent_config = {'players': flags['players']}
    self.environment = rl_env.make('Hanabi-Full', num_players=flags['players'])
    self.a1 = a1
    self.a2 = a2
    self.lenient = lenient

  def run(self):
    """Run episodes."""
    obs_stacker = run_paired_experiment.create_obs_stacker(self.environment)

    num_episodes = self.flags['num_episodes']


    hints_given = np.zeros(self.environment.players)
    hints_possible = np.zeros(self.environment.players)
    total_information_plays = np.zeros(self.environment.players)
    num_plays = np.zeros(self.environment.players)
    points_scored = np.zeros(self.environment.players)
    mistakes_made = np.zeros(self.environment.players)
    total_bombed = 0

    rewards = []
    for episode in range(num_episodes):
      episode_length, episode_return, lr, sr, num_bombed, hg, hp, tip, num_p, ps, mm = run_paired_experiment.run_episode_behavioral(self.a1, self.a2, self.lenient, self.environment, obs_stacker)
      rewards.append(episode_return)
      hints_given += hg
      hints_possible += hp
      total_information_plays += tip
      num_plays += num_p
      points_scored += ps
      mistakes_made += mm
      total_bombed += num_bombed
    return rewards, hints_given/hints_possible, total_information_plays/num_plays, points_scored/num_episodes, mistakes_made/num_episodes, float(total_bombed)/float(num_episodes)


    # # agents = [self.agent_class(self.agent_config)
    # #             for _ in range(self.flags['players'])]
    # # agents = [a1,a2]
    #   reward_since_last_action = np.zeros(self.environment.players)
    #   # if np.random.uniform() <= 0.5:
    #   #   agents = [self.a1,self.a2]
    #   # else:
    #   #   agents = [self.a2,self.a1]
    #   agents = [self.a1,self.a2]
    #   obs_stacker.reset_stack()
    #   observations = self.environment.reset()
    #   done = False
    #   episode_reward = 0
    #   first_turn = True
    #   while not done:
    #     for agent_id, agent in enumerate(agents):
    #       observation = observations['player_observations'][agent_id]
    #       if first_turn == True:
    #         # print(first_turn)
    #         # print(observation['current_player'])
    #         first_turn = False
    #       if isinstance(agent,RainbowAgent):
    #         # print(observation)
    #         # print(observation['vectorized'])
    #         # print(len(observation['vectorized']))
    #         # print(observation['legal_moves_as_int'])
    #         # print(reward_since_last_action)
    #         # print(observation['current_player'])
    #         # legal_moves = np.ones(20)
    #         # for m in observation['legal_moves_as_int']:
    #         #   legal_moves[m]=0
    #         # print(legal_moves)
    #         current_player, legal_moves, observation_vector = ( run_paired_experiment.parse_observations(observations, self.environment.num_moves(), obs_stacker))
    #         action = int(agent._select_action(observation['vectorized'],legal_moves))

    #       else:
    #         action = agent.act(observation)
    #         print('=-=-=-=-=--=-=')
    #         print('other player made action ' + str(action))
    #         print('=-=-=-=-=--=-=')
    #       if observation['current_player'] == agent_id:
    #         assert action is not None
    #         current_player_action = action
    #       else:
    #         assert action is None
    #     # Make an environment step.
    #     # # print('Agent: {} action: {}'.format(observation['current_player'],
    #     #                                     current_player_action))
    #     observations, reward, done, unused_info = self.environment.step(
    #         current_player_action)
    #     if (reward >=0 or not self.lenient):
    #       episode_reward += reward

    #   rewards.append(episode_reward)
    #   # print('Running episode: %d' % episode)
    #   # print('Reward of this episode: %d' % episode_reward)
    #   # print('Max Reward: %.3f' % max(rewards))
    #   # print('Average Reward: %.3f' % (sum(rewards)/(episode+1)))
    # # for a in agents:
    # #   a.rulebased.print_histogram()

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
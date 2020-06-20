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
"""Run methods for training a DQN agent on Atari.

Methods in this module are usually referenced by |train.py|.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import copy

import pdb

from third_party.dopamine import checkpointer
from third_party.dopamine import iteration_statistics
import dqn_agent
from rainbow_agent import RainbowAgent
import gin.tf
import rl_env
import numpy as np
import rainbow_agent
import tensorflow as tf
import random
import sys


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


from internal_discard_oldest import InternalDiscardOldest
from internal_probabilistic import InternalProbabilistic
from internal_swapped import InternalSwapped

AGENTS = [IGGIAgent, InternalAgent, OuterAgent, LegalRandomAgent, VanDenBerghAgent, FlawedAgent, PiersAgent]
AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent, 'InternalAgent': InternalAgent, 
'OuterAgent': OuterAgent,'IGGIAgent':IGGIAgent,'LegalRandomAgent':LegalRandomAgent,'FlawedAgent':FlawedAgent,
'PiersAgent':PiersAgent, 'VanDenBerghAgent':VanDenBerghAgent, 'InternalDiscardOldest': InternalDiscardOldest, 'InternalProbabilistic': InternalProbabilistic, 'InternalSwapped': InternalSwapped}

LENIENT_SCORE = True

ENSEMBLE = True
MIRROR_TRAINING_PROBABILITY = 1.0 # 1/(len(AGENTS)+1)
PROFILING = False
TOTAL_STEP_COUNT = 0
TOTAL_TIME = 0
GLOBAL_RESULTS = []

class ObservationStacker(object):
  """Class for stacking agent observations."""

  def __init__(self, history_size, observation_size, num_players):
    """Initializer for observation stacker.

    Args:
      history_size: int, number of time steps to stack.
      observation_size: int, size of observation vector on one time step.
      num_players: int, number of players.
    """
    self._history_size = history_size
    self._observation_size = observation_size
    self._num_players = num_players
    self._obs_stacks = list()
    for _ in range(0, self._num_players):
      self._obs_stacks.append(np.zeros(self._observation_size *
                                       self._history_size))

  def add_observation(self, observation, current_player):
    """Adds observation for the current player.

    Args:
      observation: observation vector for current player.
      current_player: int, current player id.
    """
    self._obs_stacks[current_player] = np.roll(self._obs_stacks[current_player],
                                               -self._observation_size)
    self._obs_stacks[current_player][(self._history_size - 1) *
                                     self._observation_size:] = observation

  def get_observation_stack(self, current_player):
    """Returns the stacked observation for current player.

    Args:
      current_player: int, current player id.
    """

    return self._obs_stacks[current_player]

  def reset_stack(self):
    """Resets the observation stacks to all zero."""

    for i in range(0, self._num_players):
      self._obs_stacks[i].fill(0.0)

  @property
  def history_size(self):
    """Returns number of steps to stack."""
    return self._history_size

  def observation_size(self):
    """Returns the size of the observation vector after history stacking."""
    return self._observation_size * self._history_size


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: A list of paths to the gin configuration files for this
      experiment.
    gin_bindings: List of gin parameter bindings to override the values in the
      config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


@gin.configurable
def create_environment(game_type='Hanabi-Full', num_players=2):
  """Creates the Hanabi environment.

  Args:
    game_type: Type of game to play. Currently the following are supported:
      Hanabi-Full: Regular game.
      Hanabi-Small: The small version of Hanabi, with 2 cards and 2 colours.
    num_players: Int, number of players to play this game.

  Returns:
    A Hanabi environment.
  """
  return rl_env.make(
      environment_name=game_type, num_players=num_players, pyhanabi_path=None)


@gin.configurable
def create_obs_stacker(environment, history_size=4):
  """Creates an observation stacker.

  Args:
    environment: environment object.
    history_size: int, number of steps to stack.

  Returns:
    An observation stacker object.
  """

  return ObservationStacker(history_size,
                            environment.vectorized_observation_shape()[0],
                            environment.players)


@gin.configurable
def create_agent(environment, obs_stacker, agent_type='DQN'):
  """Creates the Hanabi agent.

  Args:
    environment: The environment.
    obs_stacker: Observation stacker object.
    agent_type: str, type of agent to construct.

  Returns:
    An agent for playing Hanabi.

  Raises:
    ValueError: if an unknown agent type is requested.
  """
  if agent_type == 'DQN':
    return dqn_agent.DQNAgent(observation_size=obs_stacker.observation_size(),
                              num_actions=environment.num_moves(),
                              num_players=environment.players)
  elif agent_type == 'Rainbow':
    return rainbow_agent.RainbowAgent(
        observation_size=obs_stacker.observation_size(),
        num_actions=environment.num_moves(),
        num_players=environment.players)
  else:
    raise ValueError('Expected valid agent_type, got {}'.format(agent_type))


def initialize_checkpointing(agent, experiment_logger, checkpoint_dir,checkpoint_save_dir,checkpoint_version,
                             checkpoint_file_prefix='ckpt'):
  """Reloads the latest checkpoint if it exists.

  The following steps will be taken:
   - This method will first create a Checkpointer object, which will be used in
     the method and then returned to the caller for later use.
   - It will then call checkpointer.get_latest_checkpoint_number to determine
     whether there is a valid checkpoint in checkpoint_dir, and what is the
     largest file number.
   - If a valid checkpoint file is found, it will load the bundled data from
     this file and will pass it to the agent for it to reload its data.
   - If the agent is able to successfully unbundle, this method will verify that
     the unbundled data contains the keys, 'logs' and 'current_iteration'. It
     will then load the Logger's data from the bundle, and will return the
     iteration number keyed by 'current_iteration' as one of the return values
     (along with the Checkpointer object).

  Args:
    agent: The agent that will unbundle the checkpoint from checkpoint_dir.
    experiment_logger: The Logger object that will be loaded from the
      checkpoint.
    checkpoint_dir: str, the directory containing the checkpoints.
    checkpoint_file_prefix: str, the checkpoint file prefix.

  Returns:
    start_iteration: int, The iteration number to start the experiment from.
    experiment_checkpointer: The experiment checkpointer.
  """
  experiment_checkpointer = checkpointer.Checkpointer(
      checkpoint_dir, checkpoint_file_prefix)

  start_iteration = 0

  # Check if checkpoint exists. Note that the existence of checkpoint 0 means
  # that we have finished iteration 0 (so we will start from iteration 1).
  if checkpoint_version == None :
    print("Didn't enter checkpoint version, will load latest checkpoint")
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(checkpoint_dir)
  else:
    print("trying to load checkpoint version {} from checkpoint dir{}".format(checkpoint_version,checkpoint_dir))
    latest_checkpoint_version = int(checkpoint_version)
  if latest_checkpoint_version >= 0:
    dqn_dictionary = experiment_checkpointer.load_checkpoint(
        latest_checkpoint_version)
    if agent.unbundle(
        checkpoint_dir, latest_checkpoint_version, dqn_dictionary):
      assert 'logs' in dqn_dictionary
      assert 'current_iteration' in dqn_dictionary
      experiment_logger.data = dqn_dictionary['logs']
      start_iteration = dqn_dictionary['current_iteration'] + 1
      tf.logging.info('Reloaded checkpoint and will start from iteration %d',
                      start_iteration)
    else:
      sys.exit("load failed")
    #redirect checkpointer from dir to save_dir
    experiment_checkpointer = checkpointer.Checkpointer(checkpoint_save_dir, checkpoint_file_prefix)
  return start_iteration, experiment_checkpointer


def format_legal_moves(legal_moves, action_dim):
  """Returns formatted legal moves.

  This function takes a list of actions and converts it into a fixed size vector
  of size action_dim. If an action is legal, its position is set to 0 and -Inf
  otherwise.
  Ex: legal_moves = [0, 1, 3], action_dim = 5
      returns [0, 0, -Inf, 0, -Inf]

  Args:
    legal_moves: list of legal actions.
    action_dim: int, number of actions.

  Returns:
    a vector of size action_dim.
  """
  new_legal_moves = np.full(action_dim, -float('inf'))
  if legal_moves:
    new_legal_moves[legal_moves] = 0
  return new_legal_moves


def parse_observations(observations, num_actions, obs_stacker):
  """Deconstructs the rich observation data into relevant components.

  Args:
    observations: dict, containing full observations.
    num_actions: int, The number of available actions.
    obs_stacker: Observation stacker object.

  Returns:
    current_player: int, Whose turn it is.
    legal_moves: `np.array` of floats, of length num_actions, whose elements
      are -inf for indices corresponding to illegal moves and 0, for those
      corresponding to legal moves.
    observation_vector: Vectorized observation for the current player.
  """
  current_player = observations['current_player']
  current_player_observation = (
      observations['player_observations'][current_player])

  legal_moves = current_player_observation['legal_moves_as_int']
  legal_moves = format_legal_moves(legal_moves, num_actions)

  observation_vector = current_player_observation['vectorized']
  obs_stacker.add_observation(observation_vector, current_player)
  observation_vector = obs_stacker.get_observation_stack(current_player)

  return current_player, legal_moves, observation_vector



# TODO: This is a special case of run_one_episode taken directly from the base Rainbow implementation. It should be unified with the next method, possibly by solving the
# agent.act inconsistency (rule-based agents expect only observation, but Rainbow expects obs + legal action)
def run_one_episode_mirror(agent, lenient, environment, obs_stacker):
  """Runs the agent on a single game of Hanabi in self-play mode.

  Args:
    agent: Agent playing Hanabi.
    environment: The Hanabi environment.
    obs_stacker: Observation stacker object.

  Returns:
    step_number: int, number of actions in this episode.
    total_reward: float, undiscounted return for this episode.
  """
  obs_stacker.reset_stack()
  observations = environment.reset()
  current_player, legal_moves, observation_vector = (
      parse_observations(observations, environment.num_moves(), obs_stacker))
  action = agent.begin_episode(current_player, legal_moves, observation_vector)

  is_done = False
  total_reward = 0
  lenient_reward = 0
  strict_reward = 0
  bombed = False
  step_number = 0

  has_played = {current_player}

  # Keep track of per-player reward.
  reward_since_last_action = np.zeros(environment.players)

  while not is_done:
    observations, reward, is_done, _ = environment.step(action.item())

    modified_reward = max(reward, 0) if lenient else reward
    total_reward += modified_reward
    lenient_reward += max(reward,0)
    strict_reward += reward

    if reward <0:
      bombed = True

    reward_since_last_action += modified_reward

    step_number += 1
    if is_done:
      break
    current_player, legal_moves, observation_vector = (
        parse_observations(observations, environment.num_moves(), obs_stacker))
    if current_player in has_played:
      action = agent.step(reward_since_last_action[current_player],
                          current_player, legal_moves, observation_vector)
    else:
      # Each player begins the episode on their first turn (which may not be
      # the first move of the game).
      action = agent.begin_episode(current_player, legal_moves,
                                   observation_vector)
      has_played.add(current_player)

    # Reset this player's reward accumulator.
    reward_since_last_action[current_player] = 0

  agent.end_episode(reward_since_last_action)

  #tf.logging.info('EPISODE: %d %g', step_number, total_reward)
  return step_number, total_reward, lenient_reward, strict_reward, bombed

@gin.configurable
def run_episode_behavioral(my_agent, their_agent, lenient, environment, obs_stacker, display_moves = False):
  """Runs the agent on a single game of Hanabi in self-play mode.

  Args:
    agent: Agent playing Hanabi.
    environment: The Hanabi environment.
    obs_stacker: Observation stacker object.

  Returns:
    step_number: int, number of actions in this episode.
    total_reward: float, undiscounted return for this episode.
  """

  # if ensemble:
  #   agent_index = random.randint(0,len(AGENTS-1))
  #   their_agent = AGENTS[agent_index]({})
  obs_stacker.reset_stack()
  observations = environment.reset()
  current_player, legal_moves, observation_vector = (
      parse_observations(observations, environment.num_moves(), obs_stacker))
  observation = observations['player_observations'][current_player]
  num_players = observation['num_players']
  my_player_index = random.randint(0,num_players-1)
  if current_player == my_player_index:
    try:
      action = my_agent.act(observation)
    except AttributeError:
      action = my_agent.begin_episode(current_player, legal_moves, observation_vector).item()
  else:
    try:
      action = their_agent.act(observation)
    except AttributeError:
      action = their_agent.begin_episode(current_player, legal_moves,
                                   observation_vector).item()

  is_done = False
  total_reward = 0
  lenient_reward = 0
  strict_reward = 0
  bombed = False
  step_number = 0

  has_played = {current_player}

  # Keep track of per-player reward.
  reward_since_last_action = np.zeros(environment.players)

  total_step_time = 0.0
  total_env_time = 0.0
  total_agent_time = 0.0
  total_partner_time = 0.0
  total_train_time = 0.0
  total_act_time = 0.0

  start_time=time.time()

  hints_given = np.zeros(environment.players)
  hints_possible = np.zeros(environment.players)
  total_information_plays = np.zeros(environment.players)
  num_plays = np.zeros(environment.players)
  points_scored = np.zeros(environment.players)
  mistakes_made = np.zeros(environment.players)

  current_lives = 2

  while not is_done:
    if display_moves:
      if current_player == 0:
        current_agent = my_agent
      else:
        current_agent = their_agent
      print("Current player is {}".format(current_agent))
      print(observation)
      if isinstance(action,dict):
        print(action)
      else:
        print("getting move")
        print(environment.game.get_move(action))
      
    # observations, reward, is_done, _ = environment.step(action.item())
    try:
      observations, reward, is_done, _ = environment.step(action)
    except ValueError:
      observations, reward, is_done, _ = environment.step(int(action))

    if display_moves:
      print("Reward of move was {}".format(reward))
      pdb.set_trace()

    # Get behavioral metrics resulting from previus action and attribute it to the non-current player. Note that at this point we haven't updated the current player yet.
    if reward == 1:
      points_scored[(current_player+my_player_index)%environment.players] +=1
    # print(observations)
    if observations['player_observations'][current_player]['life_tokens'] < current_lives:
      mistakes_made[(current_player+my_player_index)%environment.players] +=1
    current_lives = observations['player_observations'][current_player]['life_tokens']

    modified_reward = max(reward, 0) if lenient else reward
    total_reward += modified_reward
    lenient_reward += max(reward,0)
    strict_reward += reward
    if reward <0:
      bombed = True

    reward_since_last_action += modified_reward


    step_number += 1
    if is_done:
      break
    current_player, legal_moves, observation_vector = (
        parse_observations(observations, environment.num_moves(), obs_stacker))
    observation = observations['player_observations'][current_player]

    if current_player in has_played:
      if current_player == my_player_index:
        # print("Has played and zero")
        # action = my_agent.profile_step(reward_since_last_action[current_player],
        #                   current_player, legal_moves, observation_vector).item()

        try:
          action = my_agent.act(observation)
        except AttributeError:
          action = my_agent.step(reward_since_last_action[current_player],
                          current_player, legal_moves, observation_vector)
      else:
        # print("Has played and not zero")
        try:
          action = their_agent.act(observation)
        except AttributeError:
          action = their_agent.step(reward_since_last_action[current_player],
                          current_player, legal_moves, observation_vector)
    else:
      # Each player begins the episode on their first turn (which may not be
      # the first move of the game).
      if current_player == my_player_index:
        # print("Not Has played and zero")
        try:
          action = my_agent.act(observation)
        except AttributeError:
          action = my_agent.begin_episode(current_player, legal_moves,
                                   observation_vector).item()
      else:
        # print("Not Has played and not zero")
        # print(observations)
        try:
          action = their_agent.act(observation)
        except AttributeError:
          action = their_agent.begin_episode(current_player, legal_moves,
                                   observation_vector).item()
      has_played.add(current_player)

    # print(observations)
    current_player_observation=observations['player_observations'][current_player]
    # print(current_player_observation)
    # print(observations['player_observations'][(current_player+1)%2])
    # pdb.set_trace()
    if current_player_observation['information_tokens'] >0:
      hints_possible[(current_player+my_player_index)%environment.players]+=1
    # print(action)
    try:
      if action['action_type'] == 'REVEAL_RANK' or action['action_type'] == 'REVEAL_COLOR':
        hints_given[(current_player+my_player_index)%environment.players]+=1
    except (IndexError, TypeError) as e:
      if int(action) >=10:
        hints_given[(current_player+my_player_index)%environment.players]+=1

    # Check if chosen action is play:
    play_action = False
    try:
      if action['action_type'] == 'PLAY':
        play_action = True
        num_plays[(current_player+my_player_index)%environment.players] +=1
        index = action['card_index']
    except (IndexError, TypeError) as e:
      if int(action) < 5:
        play_action = True
        num_plays[(current_player+my_player_index)%environment.players] +=1
        index = int(action)
    if play_action:
      information = 0
      # print(current_player_observation)
      # print(action)
      # print(index)
      if current_player_observation['card_knowledge'][0][index]['color'] is not None:
        information+=1
        # print ("Knew color")
      if current_player_observation['card_knowledge'][0][index]['rank'] is not None:
        information+=1
        # print ("Knew Rank")
      # pdb.set_trace()
      total_information_plays[(current_player+my_player_index)%environment.players]+=information



    # print(action)
    # print(current_player_observation['legal_moves'])
    # print(current_player_observation['legal_moves_as_int'])

    # Reset this player's reward accumulator.
    reward_since_last_action[current_player] = 0
    # pdb.set_trace()

  # Profiling
  if PROFILING:
    print("-=-=--=--=Profiling-=-=-=-=-=  ")
    print("Average  time per step = {0}".format(total_step_time/step_number))
    print("Average envornment time per step = {0}".format(total_env_time/step_number))
    print("Average agent time per step = {0}".format(2*total_agent_time/step_number))
    print("Average train time per step = {0}".format(2*total_train_time/step_number))
    print("Average act time per step = {0}".format(2*total_act_time/step_number))
    print(their_agent)
    print("Average partner time per step = {0}".format(2*total_partner_time/step_number))

  if isinstance(my_agent,RainbowAgent):
    my_agent.end_episode(reward_since_last_action)

#   tf.logging.info('EPISODE: %d %g', step_number, total_reward)
  return step_number, total_reward, lenient_reward, strict_reward, bombed, hints_given, hints_possible, total_information_plays, num_plays, points_scored, mistakes_made

def run_one_episode(my_agent, their_agent, lenient, environment, obs_stacker):
  """Runs the agent on a single game of Hanabi in self-play mode.

  Args:
    agent: Agent playing Hanabi.
    environment: The Hanabi environment.
    obs_stacker: Observation stacker object.

  Returns:
    step_number: int, number of actions in this episode.
    total_reward: float, undiscounted return for this episode.
  """

  # if ensemble:
  #   agent_index = random.randint(0,len(AGENTS-1))
  #   their_agent = AGENTS[agent_index]({})
  obs_stacker.reset_stack()
  observations = environment.reset()
  current_player, legal_moves, observation_vector = (
      parse_observations(observations, environment.num_moves(), obs_stacker))
  observation = observations['player_observations'][current_player]
  num_players = observation['num_players']
  my_player_index = random.randint(0,num_players-1)
  if current_player == my_player_index:
    action = my_agent.begin_episode(current_player, legal_moves, observation_vector).item()
  else:
    try:
      action = their_agent.act(observation)
    except AttributeError:
      action = their_agent.begin_episode(current_player, legal_moves,
                                   observation_vector).item()

  is_done = False
  total_reward = 0
  lenient_reward = 0
  strict_reward = 0
  bombed = False
  step_number = 0

  has_played = {current_player}

  # Keep track of per-player reward.
  reward_since_last_action = np.zeros(environment.players)

  total_step_time = 0.0
  total_env_time = 0.0
  total_agent_time = 0.0
  total_partner_time = 0.0
  total_train_time = 0.0
  total_act_time = 0.0

  start_time=time.time()

  while not is_done:
    total_step_time += time.time()-start_time
    start_time=time.time()

    # observations, reward, is_done, _ = environment.step(action.item())
    observations, reward, is_done, _ = environment.step(action)


    modified_reward = max(reward, 0) if lenient else reward
    total_reward += modified_reward
    lenient_reward += max(reward,0)
    strict_reward += reward
    if reward <0:
      bombed = True

    reward_since_last_action += modified_reward


    step_number += 1
    if is_done:
      break
    current_player, legal_moves, observation_vector = (
        parse_observations(observations, environment.num_moves(), obs_stacker))
    observation = observations['player_observations'][current_player]


    env_time=time.time()
    total_env_time+=env_time-start_time

    if current_player in has_played:
      if current_player == my_player_index:
        # print("Has played and zero")
        # action = my_agent.profile_step(reward_since_last_action[current_player],
        #                   current_player, legal_moves, observation_vector).item()

        t, train, act = my_agent.profile_step(reward_since_last_action[current_player],
                          current_player, legal_moves, observation_vector)
        total_train_time+=train
        total_act_time+=act
        action = t.item()
        agent_time = time.time()
        total_agent_time+= agent_time-env_time
      else:
        # print("Has played and not zero")
        try:
          action = their_agent.act(observation)
        except AttributeError:
          t, train, act = their_agent.profile_step(reward_since_last_action[current_player],
                          current_player, legal_moves, observation_vector)
          action = t.item
        partner_time = time.time()
        total_partner_time+= partner_time-env_time
    else:
      # Each player begins the episode on their first turn (which may not be
      # the first move of the game).
      if current_player == my_player_index:
        # print("Not Has played and zero")
        action = my_agent.begin_episode(current_player, legal_moves,
                                   observation_vector).item()
        agent_time = time.time()
        total_agent_time+= agent_time-env_time
      else:
        # print("Not Has played and not zero")
        # print(observations)
        try:
          action = their_agent.act(observation)
        except AttributeError:
          action = their_agent.begin_episode(current_player, legal_moves,
                                   observation_vector).item()
        partner_time = time.time()
        total_partner_time+= partner_time-env_time
      has_played.add(current_player)

    # Reset this player's reward accumulator.
    reward_since_last_action[current_player] = 0

  # Profiling
  if PROFILING:
    print("-=-=--=--=Profiling-=-=-=-=-=  ")
    print("Average  time per step = {0}".format(total_step_time/step_number))
    print("Average envornment time per step = {0}".format(total_env_time/step_number))
    print("Average agent time per step = {0}".format(2*total_agent_time/step_number))
    print("Average train time per step = {0}".format(2*total_train_time/step_number))
    print("Average act time per step = {0}".format(2*total_act_time/step_number))
    print(their_agent)
    print("Average partner time per step = {0}".format(2*total_partner_time/step_number))

  my_agent.end_episode(reward_since_last_action)

#   tf.logging.info('EPISODE: %d %g', step_number, total_reward)
  return step_number, total_reward, lenient_reward, strict_reward, bombed


def run_one_phase(my_agent, training_partners, lenient, environment, obs_stacker, min_steps, statistics,
                  run_mode_str):
  """Runs the agent/environment loop until a desired number of steps.

  Args:
    agent: Agent playing hanabi.
    environment: environment object.
    obs_stacker: Observation stacker object.
    min_steps: int, minimum number of steps to generate in this phase.
    statistics: `IterationStatistics` object which records the experimental
      results.
    run_mode_str: str, describes the run mode for this agent.

  Returns:
    The number of steps taken in this phase, the sum of returns, and the
      number of episodes performed.
  """
  step_count = 0
  num_episodes = 0
  sum_returns = 0.

  while step_count < min_steps:
    partner_name = random.choice(training_partners)
    if partner_name == 'Mirror':
      episode_length, episode_return, lr, sr, num_bombed = run_one_episode_mirror(my_agent, lenient,  environment, obs_stacker)
    else: 
      their_agent = AGENT_CLASSES[partner_name]({})
      episode_length, episode_return, lr, sr, num_bombed = run_one_episode(my_agent, their_agent, lenient, environment,
                                                     obs_stacker)
    statistics.append({
        '{}_episode_lengths'.format(run_mode_str): episode_length,
        '{}_episode_returns'.format(run_mode_str): episode_return
    })

    step_count += episode_length
    sum_returns += episode_return
    num_episodes += 1

  return step_count, sum_returns, num_episodes


@gin.configurable
def run_one_iteration(my_agent, training_partners, eval_partners, lenient, environment, obs_stacker,
                      iteration, training_steps,
                      evaluate_every_n=250,
                      num_evaluation_games=100,
                      checkpoint_dir="."):
  """Runs one iteration of agent/environment interaction.

  An iteration involves running several episodes until a certain number of
  steps are obtained.

  Args:
    agent: Agent playing hanabi.
    environment: The Hanabi environment.
    obs_stacker: Observation stacker object.
    iteration: int, current iteration number, used as a global_step.
    training_steps: int, the number of training steps to perform.
    evaluate_every_n: int, frequency of evaluation.
    num_evaluation_games: int, number of games per evaluation.

  Returns:
    A dict containing summary statistics for this iteration.
  """
  start_time = time.time()

  statistics = iteration_statistics.IterationStatistics()

  # First perform the training phase, during which the agent learns.
  my_agent.eval_mode = False
  number_steps, sum_returns, num_episodes = (
      run_one_phase(my_agent, training_partners, lenient, environment, obs_stacker, training_steps, statistics,
                    'train'))

  # Also run an evaluation phase if desired.
  if evaluate_every_n is not None and iteration % evaluate_every_n == 0:
    # episode_data = []
    my_agent.eval_mode = True
    # Collect episode data for all games.
    # for _ in range(num_evaluation_games):
    #   episode_data.append(run_one_episode(my_agent, their_agent, environment, obs_stacker, False))
    #   # TODO: Here instead of doing their agent

    with open ("{0}/eval{1}".format(checkpoint_dir,iteration), "w") as eval_file:
      for ep in eval_partners:
        training_flag = False
        if ep in training_partners:
          training_flag = True
        rewards = []
        lenient_rewards = []
        strict_rewards = []
        count_bombed = 0
        if ep == 'Mirror':
          for _ in range(num_evaluation_games):
            steps, total_reward, lenient_reward, strict_reward, bombed = run_one_episode_mirror(my_agent, lenient, environment, obs_stacker)
            rewards.append(total_reward)
            lenient_rewards.append(lenient_reward)
            strict_rewards.append(strict_reward)
            if bombed:
              count_bombed+=1
          mean = np.mean(rewards)
          mean_lenient = np.mean(lenient_rewards)
          mean_strict = np.mean(strict_rewards)
          sd = np.std(rewards)
          sd_lenient = np.std(lenient_rewards)
          sd_strict = np.std(strict_rewards)
          eval_file.write("{0} Mirror {1} {2} {3} {4} {5} {6} {7} {8}\n".format(num_evaluation_games,mean,sd, mean_lenient, sd_lenient, mean_strict, sd_strict, count_bombed, training_flag))
          print ("Played {0} games with Mirror. Average score: {1} SD: {2} Bombed: {3}".format(num_evaluation_games,mean,sd,count_bombed))


        else:
          their_agent = AGENT_CLASSES[ep]({})
          for _ in range(num_evaluation_games):
            steps, total_reward, lenient_reward, strict_reward, bombed = run_one_episode(my_agent, their_agent, lenient, environment, obs_stacker)
            rewards.append(total_reward)
            lenient_rewards.append(lenient_reward)
            strict_rewards.append(strict_reward)
            if bombed:
              count_bombed+=1
          mean = np.mean(rewards)
          mean_lenient = np.mean(lenient_rewards)
          mean_strict = np.mean(strict_rewards)
          sd = np.std(rewards)
          sd_lenient = np.std(lenient_rewards)
          sd_strict = np.std(strict_rewards)

          eval_file.write("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(num_evaluation_games,ep,mean,sd, mean_lenient, sd_lenient, mean_strict, sd_strict, count_bombed, training_flag))
          print ("Played {0} games with agent {1}. Average score: {2} SD: {3} Bombed: {4}".format(num_evaluation_games,ep,mean,sd,count_bombed))

      # other_agent = my_agent
      # rewards = []
      # for _ in range(num_evaluation_games):
      #   steps, total_reward = run_one_episode(my_agent, other_agent, environment, obs_stacker, False)
      #   rewards.append(total_reward)
      # mean = np.mean(rewards)
      # sd = np.std(rewards)
      # print ("Played {0} games in self-play. Average score: {1} SD: {2} Total steps: {3}".format(num_evaluation_games,mean,sd,steps))

      # for _ in range(num_evaluation_games):
      #   steps, total_reward = run_one_episode_mirror(my_agent, environment, obs_stacker)
      #   rewards.append(total_reward)
      # mean = np.mean(rewards)
      # sd = np.std(rewards)
      # eval_file.write("{0} Mirror {1} {2}\n".format(num_evaluation_games,mean,sd,steps))
      # print ("Played {0} games in self-play (copy). Average score: {1} SD: {2} Total steps: {3}".format(num_evaluation_games,mean,sd,steps))

  
  else:
    statistics.append({
        'eval_episode_lengths': -1,
        'eval_episode_returns': -1
    })
    
  # print(their_agent)  
    
  global TOTAL_STEP_COUNT
  global TOTAL_TIME
  global GLOBAL_RESULTS
  TOTAL_STEP_COUNT+=number_steps
  time_delta = time.time() - start_time
  average_return = sum_returns / num_episodes
  TOTAL_TIME+=time_delta
  iteration_results =[TOTAL_STEP_COUNT,TOTAL_TIME,average_return]
  GLOBAL_RESULTS.append(iteration_results)


  # for r in GLOBAL_RESULTS:
  #   print(r[0],r[1],r[2])

  print("End of iteration {0}".format(iteration))
  print("Checkpoint_dir = {0}".format(checkpoint_dir))
  print("Training partners = {0}".format(training_partners))
  print("Evaluation partners = {0}".format(eval_partners))
  print("Lenient scoring  = {0}".format(lenient))
  print("training_steps = {0}".format(training_steps))

  tf.logging.info('Average training steps per second: %.2f',
                  number_steps / time_delta)
  tf.logging.info('Average per episode return: %.2f', average_return)
  statistics.append({'average_return': average_return})

  print("=-=-=---=-=")

  return statistics.data_lists


def log_experiment(experiment_logger, iteration, statistics,
                   logging_file_prefix='log', log_every_n=1000):
  """Records the results of the current iteration.

  Args:
    experiment_logger: A `Logger` object.
    iteration: int, iteration number.
    statistics: Object containing statistics to log.
    logging_file_prefix: str, prefix to use for the log files.
    log_every_n: int, specifies logging frequency.
  """
  if iteration % log_every_n == 0:
    experiment_logger['iter{:d}'.format(iteration)] = statistics
    experiment_logger.log_to_file(logging_file_prefix, iteration)


def checkpoint_experiment(experiment_checkpointer, agent, experiment_logger,
                          iteration, checkpoint_dir, checkpoint_every_n):
  """Checkpoint experiment data.

  Args:
    experiment_checkpointer: A `Checkpointer` object.
    agent: An RL agent.
    experiment_logger: a Logger object, to include its data in the checkpoint.
    iteration: int, iteration number for checkpointing.
    checkpoint_dir: str, the directory where to save checkpoints.
    checkpoint_every_n: int, the frequency for writing checkpoints.
  """
  if iteration % checkpoint_every_n == 0:
    agent_dictionary = agent.bundle_and_checkpoint(checkpoint_dir, iteration)
    if agent_dictionary:
      agent_dictionary['current_iteration'] = iteration
      agent_dictionary['logs'] = experiment_logger.data
      experiment_checkpointer.save_checkpoint(iteration, agent_dictionary)


@gin.configurable
def run_paired_experiment(my_agent,  training_partners, eval_partners, lenient,
                   environment,
                   start_iteration,
                   obs_stacker,
                   experiment_logger,
                   experiment_checkpointer,
                   checkpoint_dir,
                   num_iterations=200,
                   training_steps=5000,
                   logging_file_prefix='log',
                   log_every_n=1000,
                   checkpoint_every_n=1):
  """Runs a full experiment, spread over multiple iterations."""
  # their_agent = InternalAgent({})
  tf.logging.info('Beginning training...')
  if num_iterations <= start_iteration:
    tf.logging.warning('num_iterations (%d) < start_iteration(%d)',
                       num_iterations, start_iteration)
    return

  for iteration in range(start_iteration, num_iterations):
    start_time = time.time()
    statistics = run_one_iteration(my_agent, training_partners, eval_partners, lenient, environment, obs_stacker, iteration,
                                   training_steps,checkpoint_dir=checkpoint_dir)
    tf.logging.info('Iteration %d took %d seconds', iteration,
                    time.time() - start_time)
    start_time = time.time()
    log_experiment(experiment_logger, iteration, statistics,
                   logging_file_prefix, log_every_n)
    tf.logging.info('Logging iteration %d took %d seconds', iteration,
                    time.time() - start_time)
    start_time = time.time()
    checkpoint_experiment(experiment_checkpointer, my_agent, experiment_logger,
                          iteration, checkpoint_dir, checkpoint_every_n)
    tf.logging.info('Checkpointing iteration %d took %d seconds', iteration,
                    time.time() - start_time)

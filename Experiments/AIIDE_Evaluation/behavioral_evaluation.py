from absl import app
from absl import flags

import numpy as np
import pdb

import gin.tf


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'my_agents', 'my_agents.txt',
    'Path to file containing agents to be evaluated')
flags.DEFINE_string(
    'their_agents', 'their_agents.txt',
    'Path to file containing agents to be used as partners')
flags.DEFINE_string('num_of_iterations','10','number of iterations for each agent')

flags.DEFINE_multi_string(
    'gin_files', [],
    'List of paths to gin configuration files (e.g.'
    '"configs/hanabi_rainbow.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1").')

from rainbow_agent import RainbowAgent

from third_party.dopamine import logger
from third_party.dopamine import checkpointer


from rulebased_agent import RulebasedAgent
from internal_agent import InternalAgent
from outer_agent import OuterAgent
from iggi_agent import IGGIAgent
from legal_random_agent import LegalRandomAgent
from flawed_agent import FlawedAgent
from piers_agent import PiersAgent
from van_den_bergh_agent import VanDenBerghAgent
from outer_agent import OuterAgent

from behavioral_runner import BehavioralRunner

import run_paired_experiment

experiment_logger = None
environment = None
obs_stacker = None
rainbow_agent = None
first = True


AGENT_CLASSES = {'IGGIAgent':IGGIAgent,'InternalAgent': InternalAgent,
'OuterAgent': OuterAgent,'LegalRandomAgent':LegalRandomAgent,'VanDenBerghAgent':VanDenBerghAgent,'FlawedAgent':FlawedAgent,
'PiersAgent':PiersAgent, 'RainbowAgent':None}

SETTINGS = {'players': 2, 'num_episodes': 100}


def get_agent_descriptors(file):
  with open(file) as f:
    lines  = f.readlines()
  return lines

def make_agent(l):

  global experiment_logger
  global environment
  global obs_stacker 
  global rainbow_agent
  global first
  l = l.strip().split()
  print(l)
  if l[0].lower() == 'rulebased':
    print("RULEBASED")
    agent = AGENT_CLASSES[l[1]](SETTINGS)
    name = l[1]
  elif l[0].lower() == 'rainbow':
    print("RAINBOW")
    base_dir = l[1]
    checkpoint_dir = base_dir+l[2]
    checkpoint_save_dir = base_dir+ 'test/checkpoints'

    checkpoint_version = l[3]


    if first:
      experiment_logger = logger.Logger('{}/logs'.format(checkpoint_save_dir))
      run_paired_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
      environment = run_paired_experiment.create_environment()
      obs_stacker = run_paired_experiment.create_obs_stacker(environment)
      rainbow_agent = run_paired_experiment.create_agent(environment, obs_stacker,'Rainbow')
      first = False

    start_iteration, experiment_checkpointer = (
        run_paired_experiment.initialize_checkpointing(rainbow_agent,
                                                experiment_logger,
                                                checkpoint_dir,
                                                checkpoint_save_dir,
                                                checkpoint_version,
                                                'ckpt'))

    name = 'Rainbow_{}'.format(l[4])
    agent = rainbow_agent
  elif l[0].lower() == 'self':
    return None, 'Self'
  return agent,name


# def make_agents(file):
#   agents = []
#   names = []
#   rainbow_count = 0
#   with open(file) as f:
#     lines = f.readlines()
#     for l in lines:
#       l = l.strip().split()
#       if l[0].lower() == 'rulebased':
#         agent = AGENT_CLASSES[l[1]](SETTINGS)
#         agents.append(agent)
#         names.append(l[1])
#       elif l[0].lower() == 'rainbow':
#         base_dir = l[1]
#         checkpoint_dir = base_dir+l[2]
#         checkpoint_save_dir = base_dir+ 'test/checkpoints'

#         checkpoint_version = l[3]

#         experiment_logger = logger.Logger('{}/logs'.format(checkpoint_save_dir))


#         run_paired_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
#         environment = run_paired_experiment.create_environment()
#         obs_stacker = run_paired_experiment.create_obs_stacker(environment)
#         agent = run_paired_experiment.create_agent(environment, obs_stacker,'Rainbow')
#         start_iteration, experiment_checkpointer = (
#             run_paired_experiment.initialize_checkpointing(agent,
#                                                     experiment_logger,
#                                                     checkpoint_dir,
#                                                     checkpoint_save_dir,
#                                                     checkpoint_version,
#                                                     'ckpt'))
 

#         agents.append(agent)
#         names.append('Rainbow{}'.format(rainbow_count))
#         rainbow_count+=1
#   return agents, names


def main(unused_argv):
  # my_agents_file = FLAGS.my_agents
  # my_agents, my_names = make_agents(my_agents_file)
  # print(my_agents)
  # their_agents_file = FLAGS.their_agents
  # their_agents, their_names = make_agents(their_agents_file)
  # print(their_agents)
  

  # results = []
  # for a1 , n1 in zip(my_agents, my_names):
  #   for a2 ,n2 in zip(their_agents, their_names):
  #     runner = BehavioralRunner(SETTINGS,a1,a2)
  #     score = np.average(runner.run())
  #     print("{} {} {}".format(n1,n2,score))
  #     results.append([n1,n2,score])
  # for r in results:
  #   print(r)


  my_agents_file = FLAGS.my_agents
  my_descriptors =  get_agent_descriptors(my_agents_file)
  print(my_descriptors)
  their_agents_file = FLAGS.their_agents
  their_descriptors =  get_agent_descriptors(their_agents_file)  

  results = []
  for d1 in my_descriptors:
    print(d1)
    a1, n1 = make_agent(d1)
    for d2 in their_descriptors:
      a2, n2 = make_agent(d2)
      if a2 is None:
        a2 = a1
      runner = BehavioralRunner(SETTINGS,a1,a2)
      rewards, communicativeness, ipp, points_scored, mistakes_made, total_bombs = runner.run()
      score = np.average(rewards)
      print("{0} {1} {2} {3} {4} {5} {6} {7}".format(n1,n2,score, communicativeness, ipp, points_scored, mistakes_made, total_bombs))
      results.append([n1,n2,score, communicativeness, ipp, points_scored, mistakes_made, total_bombs])
    a1 = None

  for r in results:
    print(r)


if __name__ == '__main__':


  app.run(main)

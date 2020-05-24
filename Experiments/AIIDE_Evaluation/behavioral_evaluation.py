from absl import app
from absl import flags

import numpy as np

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



AGENT_CLASSES = {'IGGIAgent':IGGIAgent,'InternalAgent': InternalAgent,
'OuterAgent': OuterAgent,'LegalRandomAgent':LegalRandomAgent,'VanDenBerghAgent':VanDenBerghAgent,'FlawedAgent':FlawedAgent,
'PiersAgent':PiersAgent, 'RainbowAgent':None}

SETTINGS = {'players': 2, 'num_episodes': 10}


def make_agents(file):
  agents = []
  names = []
  rainbow_count = 0
  with open(file) as f:
    lines = f.readlines()
    for l in lines:
      l = l.strip().split()
      if l[0].lower() == 'rulebased':
        agent = AGENT_CLASSES[l[1]](SETTINGS)
        agents.append(agent)
        names.append(l[1])
      elif l[0].lower() == 'rainbow':
        base_dir = l[1]
        checkpoint_dir = base_dir+l[2]
        checkpoint_save_dir = base_dir+ 'test/checkpoints'

        checkpoint_version = l[3]

        experiment_logger = logger.Logger('{}/logs'.format(checkpoint_save_dir))



        run_paired_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
        environment = run_paired_experiment.create_environment()
        obs_stacker = run_paired_experiment.create_obs_stacker(environment)
        agent = run_paired_experiment.create_agent(environment, obs_stacker,'Rainbow')
        start_iteration, experiment_checkpointer = (
            run_paired_experiment.initialize_checkpointing(agent,
                                                    experiment_logger,
                                                    checkpoint_dir,
                                                    checkpoint_save_dir,
                                                    checkpoint_version,
                                                    'ckpt'))
 

        agents.append(agent)
        names.append('Rainbow{}'.format(rainbow_count))
        rainbow_count+=1
  return agents, names


def main(unused_argv):
  my_agents_file = FLAGS.my_agents
  my_agents, my_names = make_agents(my_agents_file)
  print(my_agents)
  their_agents_file = FLAGS.their_agents
  their_agents, their_names = make_agents(their_agents_file)
  print(their_agents)
  

  results = []
  for a1 , n1 in zip(my_agents, my_names):
    for a2 ,n2 in zip(their_agents, their_names):
      runner = BehavioralRunner(SETTINGS,a1,a2)
      score = np.average(runner.run())
      print("{} {} {}".format(n1,n2,score))
      results.append([n1,n2,score])
  for r in results:
    print(r)


if __name__ == '__main__':


  app.run(main)

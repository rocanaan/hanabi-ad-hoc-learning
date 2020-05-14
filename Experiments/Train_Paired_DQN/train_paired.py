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

from third_party.dopamine import logger
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

AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent, 'InternalAgent': InternalAgent, 
'OuterAgent': OuterAgent,'IGGIAgent':IGGIAgent,'LegalRandomAgent':LegalRandomAgent,'FlawedAgent':FlawedAgent,
'PiersAgent':PiersAgent, 'VanDenBerghAgent':VanDenBerghAgent}

rulebased_names = ['RandomAgent', 'InternalAgent','OuterAgent','IGGIAgent','LegalRandomAgent','FlawedAgent','PiersAgent','VanDenBerghAgent']
all_agent_names = rulebased_names.copy()
all_agent_names.append('Mirror')

import run_paired_experiment

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
flags.DEFINE_string('training_partners',None,'Agents used as partners during training')
flags.DEFINE_string('eval_partners',None,'Agents used as partners during evaluation')
flags.DEFINE_string('lenient','True','Scoring scheme used. If true, score is maintained even if all lives are lost')

def parse_partner_names(name_string):
  if name_string=='all':
    return all_agent_names
  if name_string=='rb':
    return rulebased_names
  else:
    partners=[] 
    names=name_string.split()
    for n in names:
      if n in all_agent_names:
        partners.append(n)
      else:
        exit("Partner name {0} invalid. Valid names are \'all\', \'rb or {1}".format((n,all_agent_names)))
    return partners




def launch_experiment():
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
  

  training_partners = parse_partner_names(FLAGS.training_partners)
  eval_partners = parse_partner_names(FLAGS.eval_partners)
  assert training_partners, "Could not identify training partners"
  assert eval_partners, "Could not identigy evaluation partners"
  # their_agent = VanDenBerghAgent({})


  lenient = True
  if FLAGS.lenient!='True':
    lenient = False

  checkpoint_dir = FLAGS.checkpoint_dir
  if FLAGS.checkpoint_save_dir == None:
    checkpoint_save_dir = checkpoint_dir
    experiment_logger = logger.Logger('{}/logs'.format(FLAGS.base_dir))
  else:
    checkpoint_save_dir = '{}/checkpoints'.format(FLAGS.checkpoint_save_dir)
    experiment_logger = logger.Logger('{}/logs'.format(FLAGS.checkpoint_save_dir))
    print ("set save dir as: "+ checkpoint_save_dir)

  print("Starting training of Rainbow agent")
  print("Checkpoint_dir = {0}".format(checkpoint_dir))
  print("Training partners = {0}".format(training_partners))
  print("Evaluation partners = {0}".format(eval_partners))
  print("Lenient scoring  = {0}".format(lenient))



  start_iteration, experiment_checkpointer = (
      run_paired_experiment.initialize_checkpointing(my_agent,
                                              experiment_logger,
                                              checkpoint_dir,
                                              checkpoint_save_dir,
                                              FLAGS.checkpoint_version,
                                              FLAGS.checkpoint_file_prefix))


  run_paired_experiment.run_paired_experiment(my_agent, training_partners, eval_partners, lenient, environment, start_iteration,
                                obs_stacker,
                                experiment_logger, experiment_checkpointer,
                                checkpoint_save_dir,
                                logging_file_prefix=FLAGS.logging_file_prefix)


def main(unused_argv):
  """This main function acts as a wrapper around a gin-configurable experiment.

  Args:
    unused_argv: Arguments (unused).
  """
  launch_experiment()

if __name__ == '__main__':
  app.run(main)

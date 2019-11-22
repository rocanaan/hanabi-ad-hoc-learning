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
"""Simple Agent."""

from rl_env import Agent
from ruleset import Ruleset


class RulebasedAgent():
  """Agent that applies a simple heuristic."""

  def __init__(self,rules):
    self.rules = rules
    self.totalCalls = 0
    self.histogram = [0 for i in range(len(rules)+1)]

  def get_move(self,observation):
    if observation['current_player_offset'] == 0:
      for index, rule in enumerate(self.rules):
        action = rule(observation)
        if action is not None:
          # print(rule)
          self.histogram[index]+=1
          self.totalCalls +=1
          return action
      self.histogram[-1]+=1
      self.totalCalls +=1
      return Ruleset.legal_random(observation)
    return None
 
  def print_histogram(self):
    if self.totalCalls>0:
      print([calls/self.totalCalls for calls in self.histogram ])

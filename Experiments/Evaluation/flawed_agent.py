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
from rulebased_agent import RulebasedAgent
from ruleset import Ruleset


class FlawedAgent(Agent):
  """Agent that applies a simple heuristic."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.config = config
    # Extract max info tokens or set default to 8.
    self.max_information_tokens = config.get('information_tokens', 8)

    # self.rules = [Ruleset.play_safe_card,Ruleset.tell_playable_card_outer,Ruleset.discard_randomly,Ruleset.legal_random]
    self.rules = [Ruleset.play_safe_card,
                  Ruleset.play_probably_safe_factory(0.25),
                  Ruleset.tell_randomly,
                  Ruleset.osawa_discard,
                  Ruleset.discard_oldest_first,
                  Ruleset.discard_randomly]

    print(self.rules)

    self.rulebased = RulebasedAgent(self.rules)

  def act(self, observation):
    return self.rulebased.get_move(observation)
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
import random
import numpy as np
import pyhanabi

global colors
colors = ['R', 'Y', 'G', 'W', 'B']
# ranks = [1,2,3,4,5]
num_in_deck_by_rank = [3,2,2,2,1] # Note: rank is zero-based


# Note: depending on the object calling, card could either be a dict eg {'color':'R','rank':0} or a HanabiCard instance with c.color() and c.rank() methods
def playable_card(card, fireworks):
  if isinstance(card,pyhanabi.HanabiCard):
    card = {'color':colors[card.color()],'rank':card.rank()}

  """A card is playable if it can be placed on the fireworks pile."""
  if card['color'] == None and card['rank'] != None:
      for color in colors:
          if fireworks[color] == card['rank']:
              continue
          else:
              return False

      return True
  elif card['color'] == None or card['rank'] == None:
      return False
  else:
      return card['rank'] == fireworks[card['color']]

def useless_card(card,fireworks,max_fireworks):
  if isinstance(card,pyhanabi.HanabiCard):
    card = {'color':colors[card.color()],'rank':card.rank()}
  if card['rank'] < fireworks[card['color']]:
    return True
  if card['rank'] >= max_fireworks[card['color']]:
    return True
  return False



def get_plausible_cards(observation, player_offset, hand_index):
  card_knowledge = observation['pyhanabi'].card_knowledge()[player_offset]
  hidden_card = card_knowledge[hand_index]
  plausible_cards = []
  for color_index in range(5):
    for rank_index in range(5):
      if (hidden_card.color_plausible(color_index) and hidden_card.rank_plausible(rank_index)):
        plausible_card = pyhanabi.HanabiCard(color_index,rank_index)
        plausible_cards.append(plausible_card)
  return plausible_cards


def get_visible_cards(observation,player_offset):
  visible_cards = []
  for other_player in range (1,observation['num_players']):
    if other_player != player_offset:
      their_hand = observation['observed_hands'][other_player]
      for card in their_hand:
        visible_cards.append(card)
  for card in observation['discard_pile']:
    visible_cards.append(card)
  for color, rank in observation['fireworks'].items():
    for i in range(rank):
      card = {'color':color,'rank':i}
      visible_cards.append(card)
  return visible_cards


#This returns an array of the naive probability of each card being playable from a playable from a certain player's perspective
#This ignores conventions, and also doesn't make any inferences based on the information the current player has on their hand

def get_card_playability(observation, player_offset=0):
  visible_cards = get_visible_cards(observation,player_offset)
  # print(observation)
  # print(visible_cards
  my_hand_size = len(observation['observed_hands'][player_offset])
  playability_array = np.zeros(my_hand_size)
  for hand_index in range (my_hand_size):
    total_possibilities = 0
    playable_possibilities = 0
    plausible_cards = get_plausible_cards(observation,player_offset,hand_index)
    for plausible in plausible_cards:
      num_in_deck = num_in_deck_by_rank[plausible.rank()]
      for visible in visible_cards:
        # print(str(plausible) + " " + str(visible))
        # print(visible['color'])
        # print(plausible.color())
        # print(visible['rank'])
        # print(plausible.rank())
        if visible['color'] == colors[plausible.color()] and visible['rank'] == plausible.rank():
          num_in_deck -=1
      total_possibilities += num_in_deck
      if playable_card(plausible,observation['fireworks']):
        playable_possibilities += num_in_deck
    playability_array[hand_index] = playable_possibilities/total_possibilities
    
    # for plausible in plausible_cards:
    #   num_in_deck = num_in_deck_by_rank[plausible.rank()]
    #   possible_in_deck += num_in_deck

    #   for other_player in range (1,observation['num_players']):
    #     if other_player != player_offset
    #       their_hand = observation['observed_hands'][other_player]:
    #         for card in their_hand:
    #           if card['color'] == plausible.color() and card['rank'] == plausible.rank():
    #             possible_in_deck -=1:
      # print(num_in_deck)
      # for player in range(1,observation['num_players']):
      #   if player!= player_offset:



  # print (observation['pyhanabi'].card_knowledge())
  # print(player_hints)
  return playability_array

def get_probability_useless(observation, player_offset=0):
  visible_cards = get_visible_cards(observation,player_offset)
  # print(observation)
  my_hand_size = len(observation['observed_hands'][player_offset])
  probability_useless = np.zeros(my_hand_size)
  max_fireworks = get_max_fireworks(observation)
  for hand_index in range (my_hand_size):
    total_possibilities = 0
    useless_possibilities = 0
    plausible_cards = get_plausible_cards(observation,player_offset,hand_index)
    for plausible in plausible_cards:
      num_in_deck = num_in_deck_by_rank[plausible.rank()]
      for visible in visible_cards:
        # print(str(plausible) + " " + str(visible))
        # print(visible['color'])
        # print(plausible.color())
        # print(visible['rank'])
        # print(plausible.rank())
        if visible['color'] == colors[plausible.color()] and visible['rank'] == plausible.rank():
          num_in_deck -=1
      total_possibilities += num_in_deck
      if useless_card(plausible,observation['fireworks'],max_fireworks):
        useless_possibilities += num_in_deck
    probability_useless[hand_index] = useless_possibilities/total_possibilities
    
    # for plausible in plausible_cards:
    #   num_in_deck = num_in_deck_by_rank[plausible.rank()]
    #   possible_in_deck += num_in_deck

    #   for other_player in range (1,observation['num_players']):
    #     if other_player != player_offset
    #       their_hand = observation['observed_hands'][other_player]:
    #         for card in their_hand:
    #           if card['color'] == plausible.color() and card['rank'] == plausible.rank():
    #             possible_in_deck -=1:
      # print(num_in_deck)
      # for player in range(1,observation['num_players']):
      #   if player!= player_offset:



  # print (observation['pyhanabi'].card_knowledge())
  # print(player_hints)
  return probability_useless


# Note: Fireworks goes from 0 to 5, whereas rank goes from 0 to 4
def get_max_fireworks(observation):
  discarded_cards = {}
  max_fireworks = {'R':5,'Y':5,'G':5,'W':5,'B':5}
  for card in observation['discard_pile']:
    color = card['color']
    rank = card['rank']
    label = str(color)+str(rank)
    if label not in discarded_cards:
      discarded_cards[label] = 1
    else:
      discarded_cards[label] +=1
  for label in discarded_cards:
    color = label[0]
    rank = int(label[1])
    number_in_discard = discarded_cards[label]
    if number_in_discard >= num_in_deck_by_rank[rank]:
      if max_fireworks[color] >=rank:
        max_fireworks[color] = rank
  return max_fireworks

  #   print(label)
  #   print(card)
  # for color in colors:
  #   current_value = fireworks[color]
  #   print(current_value)
  #   max_possible = 5

class Ruleset():


  @staticmethod
  def discard_oldest_first(observation):
    if (observation['information_tokens']) < 8:
      return{'action_type': 'DISCARD', 'card_index': 0}
    return None

  #Note: this is not identical to the osawa rule implemented in the Fossgalaxy framework, as there the rule only takes into account explicitly known colors and ranks
  @staticmethod
  def osawa_discard(observation):
    if observation['information_tokens'] == 8:
      return None
    fireworks = observation['fireworks']
    max_fireworks = get_max_fireworks(observation)
    safe_to_discard = False
    for card_index, card in enumerate(observation['card_knowledge'][0]):
      color = card['color']
      rank = card['rank']
      if color is not None:
        if fireworks[color] == 5:
          return{'action_type': 'DISCARD','card_index':card_index}
      if (color is not None and rank is not None):
        if (rank<fireworks[color] or rank>=max_fireworks[color]):
          return{'action_type': 'DISCARD','card_index':card_index}
      if rank is not None:
        if rank<min(fireworks.values()):
          return{'action_type': 'DISCARD','card_index':card_index}

    for card_index in range(len(observation['observed_hands'][0])):
      plausible_cards = get_plausible_cards(observation,0,card_index)
      eventually_playable=False
      for card in plausible_cards:
        color = colors[card.color()]
        rank = card.rank()
        # if (rank>=fireworks[color] and rank<max_fireworks[color]):
        if (rank<max_fireworks[color]):
          eventually_playable =True
          break
      if not eventually_playable:
        return{'action_type': 'DISCARD','card_index':card_index}
    return None


  # Note: this rule only looks at the next player on purpose, for compatibility with the Fossgalaxy implementation. Prioritizes color
  @staticmethod
  def tell_unknown(observation):
    PLAYER_OFFSET =1
    if observation['information_tokens']>0:
      their_hand = observation['observed_hands'][PLAYER_OFFSET]
      their_knowledge = observation['card_knowledge'][PLAYER_OFFSET]
      for index, card in enumerate(their_knowledge):
        if card['color'] is None:
          return{'action_type':'REVEAL_COLOR', 'color':their_hand[index]['color'], 'target_offset':PLAYER_OFFSET}
        if card['rank'] is None:
          return{'action_type':'REVEAL_RANK', 'rank':their_hand[index]['rank'], 'target_offset':PLAYER_OFFSET}
    return None

    
  # Note: this rule only looks at the next player on purpose, for compatibility with the Fossgalaxy implementation. Prioritizes color
  @staticmethod
  def tell_randomly(observation):
    if observation['information_tokens']>0:
      PLAYER_OFFSET =1
      their_hand = observation['observed_hands'][PLAYER_OFFSET]
      card = random.choice(their_hand)
      r = random.randint(0,1)
      if (r == 0):
        return {
         'action_type': 'REVEAL_RANK',
         'rank': card['rank'],
         'target_offset': PLAYER_OFFSET
        }
      else:
        return {
         'action_type': 'REVEAL_COLOR',
         'color': card['color'],
         'target_offset': PLAYER_OFFSET
        }
    return None

  @staticmethod
  def play_safe_card(observation):
    PLAYER_OFFSET = 0
    fireworks = observation['fireworks']
    # # for card_index, hint in enumerate(observation['card_knowledge'][0]):
    # #   if playable_card(hint, fireworks):
    # #       return {'action_type': 'PLAY', 'card_index': card_index}
    # playability_vector = get_card_playability(observation)
    # card_index = np.argmax(playability_vector)
    # if playability_vector[card_index]==1:
    #   action = {'action_type': 'PLAY', 'card_index': card_index}
    #   return action

    for card_index, card in enumerate(observation['card_knowledge'][0]):
      plausible_cards = get_plausible_cards(observation,PLAYER_OFFSET,card_index)
      definetly_playable = True
      for plausible in plausible_cards:
        if not playable_card(plausible,fireworks):
          definetly_playable = False
          break
      if definetly_playable:
        action = {'action_type': 'PLAY', 'card_index': card_index}
        return action
    return None

  @staticmethod
  def play_if_certain(observation):
    PLAYER_OFFSET = 0
    fireworks = observation['fireworks']
    # # for card_index, hint in enumerate(observation['card_knowledge'][0]):
    # #   if playable_card(hint, fireworks):
    # #       return {'action_type': 'PLAY', 'card_index': card_index}
    # playability_vector = get_card_playability(observation)
    # card_index = np.argmax(playability_vector)
    # if playability_vector[card_index]==1:
    #   action = {'action_type': 'PLAY', 'card_index': card_index}
    #   return action

    for card_index, card in enumerate(observation['card_knowledge'][0]):
      color = card['color']
      rank = card['rank']
      if color is not None and rank is not None:
        if  rank == fireworks[color]:
          return{'action_type': 'PLAY', 'card_index': card_index}
    return None

  # Prioritizes Rank
  @staticmethod
  def tell_playable_card_outer(observation):
    fireworks = observation['fireworks']

    # Check if it's possible to hint a card to your colleagues.
    if observation['information_tokens'] > 0:
      # Check if there are any playable cards in the hands of the opponents.
      for player_offset in range(1, observation['num_players']):
        player_hand = observation['observed_hands'][player_offset]
        player_hints = observation['card_knowledge'][player_offset]
        # Check if the card in the hand of the opponent is playable.
        for card, hint in zip(player_hand, player_hints):
          if playable_card(card, fireworks) and hint['rank'] is None:
            return {
             'action_type': 'REVEAL_RANK',
             'rank': card['rank'],
             'target_offset': player_offset
            }
          elif playable_card(card,fireworks) and hint['color'] is None:
            return {
             'action_type': 'REVEAL_COLOR',
             'color': card['color'],
             'target_offset': player_offset
            }
    return None

  @staticmethod
  def tell_dispensable_factory(min_information_tokens=8):
    def tell_dispensable(observation):
      if (observation['information_tokens']<min_information_tokens):
        fireworks = observation['fireworks']

        # Check if it's possible to hint a card to your colleagues.
        if observation['information_tokens'] > 0:
          # Check if there are any playable cards in the hands of the opponents.
          for player_offset in range(1, observation['num_players']):
            player_hand = observation['observed_hands'][player_offset]
            player_hints = observation['card_knowledge'][player_offset]
            # Check if the card in the hand of the opponent is playable.
            for card, hint in zip(player_hand, player_hints):
              color = card['color']
              rank = card['rank']
              known_color = hint['color']
              known_rank = hint['rank']
              if known_color is None and fireworks[color] == 5:
                return {'action_type':'REVEAL_COLOR','color':color,'target_offset':player_offset}
              if known_rank is None and rank < min(fireworks.values()):
                return {'action_type':'REVEAL_RANK','rank':rank,'target_offset':player_offset}
              if rank < fireworks[color]:
                if known_color is None and known_rank is not None:
                   return {'action_type':'REVEAL_COLOR','color':color,'target_offset':player_offset}
                if known_color is not None and known_rank is None:
                    return {'action_type':'REVEAL_RANK','rank':rank,'target_offset':player_offset}
      return None
    return tell_dispensable

  # As far as I can tell, this is functinally identical to Tell Playable Outer
  @staticmethod
  def tell_anyone_useful_card(observation):
    return Ruleset.tell_playable_card_outer(observation)

  @staticmethod
  def tell_anyone_useless_card(observation):
    fireworks = observation['fireworks']
    if observation['information_tokens']>1:
      max_fireworks = get_max_fireworks(observation)
      for player_offset in range(1, observation['num_players']):
        player_hand = observation['observed_hands'][player_offset]
        player_hints = observation['card_knowledge'][player_offset]
        for card, hint in zip(player_hand, player_hints):
          if useless_card(card,fireworks,max_fireworks):
            if hint['color'] is None:
              return {'action_type':'REVEAL_COLOR','color':card['color'],'target_offset':player_offset}
            if hint['rank'] is None:
              return {'action_type':'REVEAL_RANK','rank':card['rank'],'target_offset':player_offset}
    return None

  # Note: this follows the version of the rule that's used on VanDenBergh, which does not take into account whether or not they already know that information
  @staticmethod
  def tell_most_information(observation):
    fireworks = observation['fireworks']
    if observation['information_tokens']>1:
      max_fireworks = get_max_fireworks(observation)
      max_affected = -1
      best_action = None
      for player_offset in range(1, observation['num_players']):
        player_hand = observation['observed_hands'][player_offset]
        player_hints = observation['card_knowledge'][player_offset]
        for card, hint in zip(player_hand, player_hints):
          affected_colors = 0
          affected_ranks = 0
          for other_card in player_hand:
            if card['color'] == other_card['color']:
              affected_colors+=1
            if card['rank']  == other_card['rank']:
              affected_ranks+=1
          if affected_colors > max_affected:
            max_affected = affected_colors
            best_action = {'action_type':'REVEAL_COLOR','color':card['color'],'target_offset':player_offset}
          if affected_ranks > max_affected:
            max_affected = affected_ranks
            best_action = {'action_type':'REVEAL_RANK','rank':card['rank'],'target_offset':player_offset}
    return None





  #Does not take into account what information the other player has into account, and decides whether to hint rank or color randomly
  @staticmethod
  def tell_playable_card(observation):
    fireworks = observation['fireworks']

    # Check if it's possible to hint a card to your colleagues.
    if observation['information_tokens'] > 0:
      # Check if there are any playable cards in the hands of the opponents.
      for player_offset in range(1, observation['num_players']):
        player_hand = observation['observed_hands'][player_offset]
        # Check if the card in the hand of the opponent is playable.
        for card in player_hand:
          if playable_card(card, fireworks):
            r = random.randint(0,1)
            if (r == 0):
              return {
               'action_type': 'REVEAL_RANK',
               'rank': card['rank'],
               'target_offset': player_offset
              }
            else:
              return {
               'action_type': 'REVEAL_COLOR',
               'color': card['color'],
               'target_offset': player_offset
              }
    return None


  @staticmethod
  def legal_random(observation):
    """Act based on an observation."""
    if observation['current_player_offset'] == 0:
      action = random.choice(observation['legal_moves'])
      return action
    else:
      return None

  @staticmethod
  def discard_randomly(observation):
    if observation['information_tokens'] < 8:
      player_offset = 0
      hand = observation['observed_hands'][player_offset]
      hand_size = len(hand)
      discard_index = random.randint(0,hand_size-1)
      return {'action_type': 'DISCARD', 'card_index': discard_index}
    return None

  @staticmethod
  def play_probably_safe_factory(treshold = 0.95, require_extra_lives = False):
    def play_probably_safe_treshold(observation):
      playability_vector = get_card_playability(observation)
      card_index = np.argmax(playability_vector)
      if not require_extra_lives or observation['life_tokens'] >1:
        if playability_vector[card_index]>=treshold:
          action = {'action_type': 'PLAY', 'card_index': card_index}
          return action
      return None

    return play_probably_safe_treshold

  @staticmethod
  def discard_probably_useless_factory(treshold = 0.75):
    def play_probably_useless_treshold(observation):
      if observation['information_tokens'] < 8: 
        probability_useless = get_probability_useless(observation)
        # print("probability useless" +str(probability_useless))
        card_index = np.argmax(probability_useless)
        if probability_useless[card_index]>=treshold:
          action = {'action_type': 'DISCARD', 'card_index': card_index}
          return action
      return None

    return play_probably_useless_treshold

  # "Hail Mary" rule used by agent Piers
  @staticmethod
  def hail_mary(observation):
    if (observation['deck_size'] == 0 and observation['life_tokens'] > 1):
      return Ruleset.play_probably_safe_factory(0.0)(observation)

"""
This example produces networks that can remember a variable-length sequence of bits. It is
intentionally very (overly?) simplistic just to show the usage of the NEAT library. However,
if you come up with a more interesting or impressive example, please submit a pull request!

This example also demonstrates the use of a custom activation function.
"""

from __future__ import print_function

#import math
import os
import random

import neat
import visualize

import numpy as np
import pyhanabi

import run_experiment


# Maximum length of the test sequence.
max_inputs = 2
# Maximum number of ignored inputs
max_ignore = 2
# Number of random examples each network is tested against.
num_tests = 2 ** (max_inputs+max_ignore+1)
# Number of games per network evaluation
num_games = 10



def test_network(net, input_sequence, num_ignore):
    # Feed input bits to the network with the record bit set enabled and play bit disabled.
    net.reset()
    for s in input_sequence:
        inputs = [random.uniform(0.0, 1.0) for i in range(2632)]
        net.activate(inputs)

    # Feed a random number of random inputs to be ignored, with both
    # record and play bits disabled.
    for _ in range(num_ignore):
        inputs = [random.uniform(0.0, 1.0) for i in range(2632)]
        net.activate(inputs)

    # Enable the play bit and get network output.
    outputs = []
    for s in input_sequence:
        inputs = [random.uniform(0.0, 1.0) for i in range(2632)]
        outputs.append(net.activate(inputs))
        print (net.activate(inputs))

    return outputs


def eval_genome(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)

    error = 0.0
    aggregate_score = 0
    environment = run_experiment.create_environment()
    obs_stacker = run_experiment.create_obs_stacker(environment)
    for _ in range(num_games):
        score = play_game(net, environment, obs_stacker)
        aggregate_score += score
        # num_inputs = random.randint(1, max_inputs)
        # num_ignore = random.randint(0, max_ignore)

        # # Random sequence of inputs.
        # seq = [random.choice((0.0, 1.0)) for _ in range(num_inputs)]

        # net.reset()
        # outputs = test_network(net, seq, num_ignore)

        # # Enable the play bit and get network output.
        # for i, o in zip(seq, outputs):
        #     error += (o[0] - i) ** 2

    return (aggregate_score/num_games)/25

def play_game(agent, environment, obs_stacker):

  obs_stacker.reset_stack()
  observations = environment.reset()
  current_player, legal_moves, observation_vector = (
      run_experiment.parse_observations(observations, environment.num_moves(), obs_stacker))


  action = get_action(agent, legal_moves, observation_vector)

  is_done = False
  total_reward = 0
  step_number = 0

  has_played = {current_player}

  # Keep track of per-player reward.
  reward_since_last_action = np.zeros(environment.players)

  total_reward = 0

  score = 0

  while not is_done:
    observations, reward, is_done, _ = environment.step(action.item())
    player_observations = observations['player_observations'][0]
    fireworks = player_observations['fireworks']
    score = int(fireworks['R'])+int(fireworks['Y'])+int(fireworks['G'])+int(fireworks['W'])+int(fireworks['B'])
    # print(type(observations))
    # fireworks = observations['player_observations'][0]['Fireworks']
    # print("FIIIIIREWWWOOOORKS")
    # print(fireworks)
    # score = fireworks['R']+fireworks['Y']+fireworks['G']+fireworks['W']+fireworks['B']
    # print(score)
    # print(observations)
    # print(observations['player_observations']['fireworks'])
    # print(observations["Fireworks"]["R"])

    # score=observations["Fireworks"]["R"] + observations["Fireworks"]["Y"] + observations["Fireworks"]["B"] + observations["Fireworks"]["W"] + observations["Fireworks"]["G"]
    total_reward += reward


    step_number += 1
    if is_done:
      # print(observations)
      # print ("SCOOOORE")
      # print(score)
      break
    current_player, legal_moves, observation_vector = (
        run_experiment.parse_observations(observations, environment.num_moves(), obs_stacker))
    if current_player in has_played:
      action = get_action(agent, legal_moves, observation_vector)
    else:
      # Each player begins the episode on their first turn (which may not be
      # the first move of the game).
      action = get_action(agent, legal_moves, observation_vector)
      has_played.add(current_player)



  # agent.end_episode(reward_since_last_action)

  # tf.logging.info('EPISODE: %d %g', step_number, total_reward)
  # return step_number, total_reward
  # print(total_reward)
  # print (score)
  return score

def get_action(agent, legal_moves, observation_vector):
    # legal_action_indices = np.where(legal_moves == 0.0)
    # return np.random.choice(legal_action_indices[0])
    activation = agent.activate(observation_vector)
    activation = np.add(activation,legal_moves)
    action = np.argmax(activation)
    # print(action)
    return action


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():

    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(4, eval_genome)
    winner = pop.run(pe.evaluate, 10000)

    # Show output of the most fit genome against a random input.
    print('\nBest genome:\n{!s}'.format(winner))
    # print('\nOutput:')
    # winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    # for n in range(num_tests):
    #     print('\nRun {0} output:'.format(n))

    #     num_inputs = random.randint(1, max_inputs)
    #     num_ignore = random.randint(0, max_ignore)

    #     seq = [random.choice((0.0, 1.0)) for _ in range(num_inputs)]
    #     winner_net.reset()
    #     outputs = test_network(winner_net, seq, num_ignore)

    #     correct = True
    #     for i, o in zip(seq, outputs):
    #         print("\texpected {0:1.5f} got {1:1.5f}".format(i, o[0]))
    #         correct = correct and round(o[0]) == i
    #     print("OK" if correct else "FAIL")

    # node_names = {-1: 'input', -2: 'record', -3: 'play', 0: 'output'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)

if __name__ == '__main__':
    run()

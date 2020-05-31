run_paired_experiment.num_iterations =100 in hanabi_rainbow.gin means the number of iterations for rainbow agents.
'num_episodes': 100 in evaluate_paired.py means the number of iterations for other agents.
run 'run_from_checkpoint_all' to get a csv file of mean and std of scores
When adding more agents, besides moving py file into this repo, remember to add agent name in AGENT_CLASSES in evaluate_paired.py.
Currently two rainbow agents read from the same checkpoint. Will enable reading from different checkpoints later.

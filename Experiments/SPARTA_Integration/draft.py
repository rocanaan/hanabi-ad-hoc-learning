


sparta_state = get_SPARTA_State()

hle_state= convert_sparta_hle()

json.write(path,sparta_state)




######


sparta_state = get_SPARTA_State()

hle_state= convert_sparta_hle()

name = "rulebased IGGIAgent"
settings = {'players': 2, 'num_episodes': num_episodes}

hle_agent  = make_agent(name,settings)

hle_action = hle_agent.act(hle_statehl) #retunrs a dict like {'action_type': 'PLAY', 'card_index': 3}

sparta.apply_action(convert(hle_action))
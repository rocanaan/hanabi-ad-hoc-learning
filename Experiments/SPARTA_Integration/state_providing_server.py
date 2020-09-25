
import os
import json
from shutil import copyfile
import time


if __name__ == '__main__':

  state_bank = 'state_bank'
  files = []
  for (dirpath, dirnames, filenames) in os.walk(state_bank):
      for f in filenames:
        files.append(dirpath+os.path.sep+f)

  actions_path = 'actions'

  output_path = 'states'
  i= 0
  for f in files:
    src = f
    dst = output_path + os.path.sep + 'state' + str(i) + '.json'
    copyfile(src, dst)

    print("Copied file " + src + " to " + dst)


    action = None
    while action is None:
      action_dir = os.listdir(actions_path) 
      found = False
      for a  in action_dir:
        if "action" in  a:
          found = True
          print("found")
          in_file = actions_path + os.path.sep + a
          with open(in_file) as j:
            data = json.load(j)
          action = str(data['action'])
          print("read action: " + action)
          print("removing file " + in_file)
          os.remove(in_file)
          break
        if not found:
          print ("Waiting for action")
          time.sleep(1)



  #     action = None

  #     while action = None:
  #       try:


  # print("Starting")

  # num_episodes = int(FLAGS.num_of_iterations)

  # name = "rulebased IGGIAgent"
  # settings = {'players': 2, 'num_episodes': num_episodes}
  # a1,n1 = make_agent(name,settings)

  # import os
  # import json

  # path = 'states'
  # print(path)
  # for (dirpath, dirnames, filenames) in os.walk(path):
  #   for f in filenames:
  #     files.append(dirpath+os.path.sep+f)

  # for f in files:
  #   with open(f) as j:
  #     data = json.load(j)
  #     print(data)
  #     print(dir(a1))
  #     action = a1.act(data)
  #     print(action)
  

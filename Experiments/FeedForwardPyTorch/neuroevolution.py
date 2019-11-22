import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import ffnn




def main():
    net = ffnn.Net()
    net.print_parameters()
    # print(net)
    # net.print_parameters()
    # print("!!!!!!!!!!!!!")
    # net.print_parameters()


    # input = torch.randn(10)
 


    num_parameters = net.num_parameters()
    print(num_parameters)
    new_parameters = [i/num_parameters for i in range(num_parameters)]
    net.set_parameters(new_parameters,num_parameters)
    print("-==-=-=-=-==-=-")

    net.print_parameters()

                # # # print(net)
                # net.print_parameters()
    # out = net(input)
    # print(out)

    # input = torch.randn(10)
    # out = net(input)
    # print(out)


    # input = torch.randn(10)
    # out = net(input)
    # print(out)

    # l=[]

if __name__ == '__main__':
    main()

# net = Net()
# print(net)

# input = torch.randn(10)
# out = net(input)
# print(out)


# input = torch.randn(10)
# out = net(input)
# print(out)

# l=[]

# model_parameters = filter(lambda p: p.requires_grad, net.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print(params)

# newparams = [i for i in range(params)]
# # print(newparams)

# index = 0
# numparams = 0
# for p in net.parameters():
#     start = index
#     layersize = np.prod(p.size())
#     new_layer_params = [(newparams[start+j])/params for j in range(layersize)]
#     print(new_layer_params)
#     index+=layersize
#     numparams+=layersize
#     print(p)
#     p = Parameter(torch.tensor(new_layer_params))
#     print(p)

# print(numparams)
# print(params)
# print(net.num_parameters())
# print(net.num_flat_features(input))

    # layerparams=1
    # print(p.shape)
    # s = p.shape
    # for dim in s:
    #     layerparams*=dim
    # i = 0
    # t = torch.ones(p.shape)
    # for v in t:
    #     v = i
    #     i = i+1
    #     p = Parameter(t)
    # print(p)
  # see tuple

# print("Printing parameters")
# numparams = 0
# for p in net.parameters():
#     layerparams=1
#     print(p.shape)
#     s = p.shape
#     for dim in s:
#         layerparams*=dim
#     i = 0
#     t = torch.ones(p.shape)
#     for v in t:
#         v = i
#         i = i+1
#         p = Parameter(t)
#     print(p)
#     numparams+=layerparams
# 	# see tuple

# print(numparams)






# print(l)


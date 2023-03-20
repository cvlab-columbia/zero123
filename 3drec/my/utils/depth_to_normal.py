#!/usr/bin/env python3
import torch
from torch.functional import norm
#from torch._C import double
import torch.nn as nn
import torch.nn.functional as F

import timeit
"""
Calculates surface normal maps given dense depth maps. Uses Simple Haar feature kernels to calculate surface gradients
"""

class SurfaceNet(nn.Module):

    def __init__(self):
        super(SurfaceNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.convDelYDelZ = nn.Conv2d(1, 1, 3)
        self.convDelXDelZ = nn.Conv2d(1, 1, 3)
        # if torch.cuda.is_available():
        # #     dev = "cuda:0"
        # # else:
        # #     dev = "cpu" 
        # # self.device = torch.device(dev)
        # # print("dev!!!", dev)  

    def forward(self, x):
        #start = timeit.default_timer()
        #x = x.to(self.device)
        nb_channels = 1#x.shape[1]
        h, w = x.shape[-2:]

        device = list(self.convDelXDelZ.parameters())[0].device

        delzdelxkernel = torch.tensor([[0.00000, 0.00000, 0.00000],
                                        [-1.00000, 0.00000, 1.00000],
                                        [0.00000, 0.00000, 0.00000]])
        delzdelxkernel = delzdelxkernel.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1).to(device)
        delzdelx = F.conv2d(x, delzdelxkernel)

        delzdelykernel = torch.tensor([[0.00000, -1.00000, 0.00000],
                                        [0.00000, 0.00000, 0.00000],
                                        [0.0000, 1.00000, 0.00000]])
        delzdelykernel = delzdelykernel.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1).to(device)

        delzdely = F.conv2d(x, delzdelykernel)

        delzdelz = torch.ones(delzdely.shape, dtype=torch.float64).to(device)
        #print('kernel',delzdelx.shape)
        surface_norm = torch.stack((-delzdelx,-delzdely, delzdelz),2)
        surface_norm = torch.div(surface_norm,  norm(surface_norm, dim=2)[:,:,None,:,:])
        # * normal vector space from [-1.00,1.00] to [0,255] for visualization processes
        #surface_norm_viz = torch.mul(torch.add(surface_norm, 1.00000),127 )
        
        #end = timeit.default_timer()
        #print("torch method time", end-start)
        return surface_norm#_viz
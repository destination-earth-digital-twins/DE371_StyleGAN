#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:45:24 2023

@author: brochetc

scattering loss for pytorch

"""

import torch.nn as nn

import torch.nn.functional as F

from kymatio.torch import Scattering2D



class ScatteringLoss(nn.Module) : 
    def __init__(self, J=1, order=1, height=128, width=128, device='cpu') :
        
        super().__init__()
        
        self.J = J
        self.max_order = order
        self.shape = (height, width)
        
        self.scattering = Scattering2D(J=J, max_order=order, shape=self.shape).to(device)
        
    def forward(self, x , y) :
        
        Scx = self.scattering(x)
        Scy = self.scattering(y)
        
        loss  =  F.mse_loss(input=Scx, target=Scy)
        
        return loss
        
    
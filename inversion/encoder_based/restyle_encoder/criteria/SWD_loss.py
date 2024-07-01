#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:39:22 2022

@author: brochetc


Sliced Wasserstein Distance API and functions
"""


import numpy as np
import torch
import torch.nn as nn


def get_descriptors_for_minibatch_torch(minibatch, nhood_size, nhoods_per_image):
    S = minibatch.shape # (minibatch, channel, height, width)
    assert len(S) == 4
    N = nhoods_per_image * S[0] //2
    H = nhood_size // 2
    nhood, chan, x, y = np.ogrid[0:N, 0:S[1], -H:H+1, -H:H+1]
    img = nhood // nhoods_per_image
    x = x + np.random.randint(H, S[3] - H, size=(N, 1, 1, 1))
    y = y + np.random.randint(H, S[2] - H, size=(N, 1, 1, 1))
    idx = ((img * S[1] + chan) * S[2] + y) * S[3] + x
    
    ddi = torch.flatten(torch.tensor(idx))
    dd_real = torch.flatten(minibatch[:S[0]//2])
    
    desc_fin_real = dd_real[ddi].view(N,S[1], nhood_size, nhood_size)
    
    dd_fake = torch.flatten(minibatch[S[0]//2:])
    
    desc_fin_fake = dd_fake[ddi].view(N,S[1], nhood_size, nhood_size)
    
    return torch.cat((desc_fin_real, desc_fin_fake), dim = 0)

#----------------------------------------------------------------------------

def finalize_descriptors_torch(desc):
    if isinstance(desc, list):
        desc = torch.cat(desc, dim=0)
    assert desc.ndim == 4 # (neighborhood, channel, height, width)
    with torch.no_grad() :
        Mean = desc.mean(dim=(0, 2, 3), keepdims=True)
        Std = desc.std(dim=(0, 2, 3), keepdims=True)
    
    desc = desc - Mean # normalizing on each channel
    desc = desc / Std  # similar to batch+instance norm

    desc = desc.reshape(desc.shape[0], -1) #reshaping
    return desc

#----------------------------------------------------------------------------

def sliced_wasserstein_torch(A,B, dir_repeats, dirs_per_repeat, device):
    
    assert A.ndim==2 and A.shape == B.shape
    
    res = []
    
    for repeat in range(dir_repeats) :
        
        dirs = torch.empty((A.shape[1], dirs_per_repeat)).normal_().to(device)
        dirs.div_(torch.linalg.norm(dirs, dim =0, keepdims = True))
        
        
        proj_A = torch.sort(torch.matmul(A, dirs), dim =0)[0]
        proj_B = torch.sort(torch.matmul(B, dirs), dim =0)[0]
        
        dists = torch.abs(proj_A - proj_B)
        
        res.append(dists.mean().unsqueeze(0))

    return torch.cat(res, dim=0).mean().unsqueeze(0)

#----------------------------------------------------------------------------

class SwdLoss:
    def __init__(self, image_shape, device='cpu'):
        self.nhood_size         = 7
        self.nhoods_per_image   = 128
        self.dir_repeats        = 4
        self.dirs_per_repeat    = 128
        self.resolutions = []

        res = image_shape[1]
        while res >= 16:
            self.resolutions.append(res)
            res //= 2
            
        gaussian_filter = np.float32([
        [1, 4,  6,  4,  1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4,  6,  4,  1]]) / 256.0
        
        filter_weights = torch.tensor(gaussian_filter).view(1,1,5,5).repeat(3,1,1,1)
        
        self.pad = nn.ReflectionPad2d(2)
        
        self.conv = nn.Conv2d(3,3, kernel_size = (5,5), groups = 3, bias = False)
        self.conv_up = nn.ConvTranspose2d(3,3, kernel_size = (5,5), groups = 3,
                                          stride = 2, bias = False, padding = (1,1))
        self.down = nn.AvgPool2d(2, stride = 2)
        
        with torch.no_grad():
            self.conv.weight = nn.Parameter(filter_weights)
            self.conv_up.weight = nn.Parameter(filter_weights)
            
        self.conv.to(device)
        self.conv_up.to(device)
        self.down.to(device)
        
        self.device = device
        

    def get_metric_names(self):
        return ['SWDx1e3_%d' % res for res in self.resolutions] + ['SWDx1e3_avg']

    def get_metric_formatting(self):
        return ['%-13.4f'] * len(self.get_metric_names())

    def begin(self, mode):
        assert mode in ['warmup', 'reals', 'fakes']
        self.desc_real = [[] for res in self.resolutions]
        self.desc_fake = [[] for res in self.resolutions]
    
    def pyr_down_torch(self, minibatch) :
        
        assert minibatch.ndim==4
        
        out = self.down(self.conv(self.pad(minibatch)))
        
        return out 


    def pyr_up_convT(self, minibatch) :
        
        S = minibatch.shape
        assert minibatch.ndim == 4
    
        out = self.conv_up(minibatch) * 4.0
        
        out = out[:,:,:2*S[2], : 2*S[3]]
        
        return out
    
    def generate_laplacian_pyramid_torch(self, minibatch, num_levels) :
        
        pyramid = [minibatch]
        for i in range(1, num_levels):
            pyramid.append(self.pyr_down_torch(pyramid[-1]))
            pyramid[-2] = pyramid[-2] - self.pyr_up_convT(pyramid[-1])
        return pyramid
    
    def reconstruct_laplacian_pyramid_torch(self,pyramid):
        minibatch = pyramid[-1]
        for level in pyramid[-2::-1]:
            minibatch = self.pyr_up_convT(minibatch) + level
        return minibatch
    

    def feed(self, minibatch):
        
        minibatch = minibatch.to(self.device)
        
        for lod, level in enumerate(self.generate_laplacian_pyramid_torch(minibatch,
                                                                          len(self.resolutions))):
            desc = get_descriptors_for_minibatch_torch(level, self.nhood_size,
                                                       self.nhoods_per_image)
            self.desc_real[lod] = finalize_descriptors_torch(desc[:desc.shape[0]//2])
            self.desc_fake[lod] = finalize_descriptors_torch(desc[desc.shape[0]//2:])

    def end(self):
        
        dist = [sliced_wasserstein_torch(dreal, dfake, self.dir_repeats, 
                                         self.dirs_per_repeat, self.device) \
                for dreal, dfake in zip(self.desc_real, self.desc_fake)]
        
        #dist = [d * 1e3 for d in dist] # multiply by 10^3
        out = torch.cat(dist, dim = 0)
        return out.mean()
    
    def End2End(self, real, fakes):
        
        self.begin('fakes')
        
        self.feed(torch.cat((real, fakes), dim=0))

        return self.end()
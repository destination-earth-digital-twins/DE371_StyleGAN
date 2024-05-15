#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 08:38:55 2023

@author: brochetc


log transform

"""


import numpy as np


lifiles = ['/scratch/mrmn/gandonb/data/crop_processed_giga/{}.npy'.format(i) for i in range(1,26)]


Mean = np.zeros((5,))

Mins = np.zeros((5,))
Maxs = np.zeros((5,))


for i,f in enumerate(lifiles) :
    
    print(f)
    
    data = np.load(f).astype(np.float32)
    
    data[:,0] = np.log(1 + data[:,0])
    
    Mean = Mean + data.mean(axis=(0,2,3))
    
    mins = np.min(data, axis=(0,2,3))
    
    maxs = np.max(data, axis=(0,2,3))
    
    if i==0 :
        Mins = mins
        Maxs = maxs
    else:
        Mins = np.minimum(mins, Mins)
        Maxs = np.maximum(maxs, Maxs)
    
    print(Mean, Mins, Maxs)
    
Mean = Mean / len(lifiles)

Maxs_m = Maxs - Mean
Mins_m = np.abs(Mins - Mean)


print(Maxs_m, Mins_m)

Maxs_abs = np.maximum(Maxs_m, Mins_m)

print(Maxs_abs, Mean)
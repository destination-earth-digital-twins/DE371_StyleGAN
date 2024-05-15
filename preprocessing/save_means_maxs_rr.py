#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:05:00 2023

@author: brochetc
"""

import numpy as np


Means = np.array([ 3.93358686e-02,  8.16572049e-01, -7.34112884e-01,  2.87447349e+02,
  3.15123922e-02]) 

Maxs = np.array([5.74386344, 36.23238196, 42.47807584, 47.46977905,  0.96848761])


np.save('./mean_log_rr_imp.npy', Means)
np.save('./max_log_rr_imp.npy', Maxs)

lifiles = ['/scratch/mrmn/gandonb/data/crop_processed_giga/{}.npy'.format(i) for i in range(1,26)]


for i,f in enumerate(lifiles) :
    
    print(f)
    
    data = np.load(f).astype(np.float32)
    
    data[:,0] = np.log(1 + data[:,0])
    
    data_sc = (data - Means.reshape(1,5,1,1))/Maxs.reshape(1,5,1,1)
    
    print(data_sc.mean(axis=(0,2,3)), data_sc.min(axis=(0,2,3)), data_sc.max(axis=(0,2,3)))
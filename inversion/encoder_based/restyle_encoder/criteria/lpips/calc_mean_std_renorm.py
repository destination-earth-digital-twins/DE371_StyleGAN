#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:27:12 2023

@author: brochetc
"""

import numpy as np
from glob import glob
data_dir = '/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_done/'

def load_batch(data_dir, N_samples):
    
    lifiles = glob(data_dir + '_sample*.npy')
    
    BigMat = np.zeros((N_samples,3,128,128))
    
    for i,file in enumerate(lifiles) :
        if i%1000==0 : print(i)
        BigMat[i] = np.load(file)[1:4,78:206,55:183]
    
    return BigMat


def calc_mean(BigMat) :
    
    return BigMat.mean(axis = (0,2,3))

def calc_std(BigMat):
    
    return BigMat.std(axis = (0,2,3))

def renorm(BigMat, Mean, Maxs, scale) :
    
    return scale * (BigMat - Mean) / Maxs

if __name__=="__main__" :
    
    BigMat = load_batch(data_dir,66048)
    
    Means = np.load(data_dir + 'mean_with_orog.npy')[1:4].reshape(1,3,1,1)
    
    Maxs = np.load(data_dir + 'max_with_orog.npy')[1:4].reshape(1,3,1,1)
    
    renormMat = renorm(BigMat, Means, Maxs, 0.95)
    
    mean_renorm = calc_mean(renormMat)
    std_renorm = calc_std(renormMat)
    
    print(mean_renorm, std_renorm)
    
    
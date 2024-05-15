#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:25:42 2023

@author: brochetc

Computation of log-likelihood (~entropy) for ensembles

"""
from math import log, sqrt, pi
import numpy as np
#from numba import gu_vectorize

def compute_log_likelihood(x, cond, sigma):
    """
    x :  C x H x W map
    cond : N x C x H x W map
    sigma : C x H x W map, dispersion of data on each variable-pixel point
    
    compute -log(p(x|cond)) (as C x H x W map) with the hypothesis of a sum of 
    Gaussian conditional distributions with same spread
    
    
    """
    N = cond.shape[0]
    
    #if sigma.min()==0 :
    #    print(sigma.min())

    X = np.expand_dims(x,0)
    
    S = np.exp(- 0.5 * ((X - cond) / sigma)**2) / (N * np.sqrt(2 * pi) * sigma)
    
    #print(S.shape, S.max(), S.min(), sigma.min(), sigma.max())
    
    S = S.sum(axis=0)
    
    Sneg = sigma[S==0].shape[0]
    #if Sneg.shape[0]>0 :
    #    for j in range(Sneg.shape[0]):  
    #        print(Sneg[j] * 4)
  
    
    res = - np.log(S+1e-8)
    
    return res , Sneg

def ensemble_log_likelihood(cond, X) :
    """
    X :  N0 x C x H x W map # data on which to evaluate entropy
    cond : N x C x H x W map # condition on whch to evaluate distribution
    sigma : C x H x W map, dispersion of data on each variable-pixel point
    
    compute -log(p(X|cond)) (as C x H x W map) with the hypothesis of a sum of of a sum of 
    Gaussian conditional distributions with same spread and independent elements
    X1, .., X_N0
    
    """
    
    Cm, Cstd = cond.mean(axis=0), cond.std(axis=0)
    
    cond = (cond - Cm) / Cstd
    
    X = (X - Cm) / Cstd

    N0, C, H, W = X.shape
    
    N =  cond.shape[0]
    
    sigma = np.std(cond, axis=0) * 1.06 / (N**(0.2))

    ell = np.zeros((C, H, W)).astype(np.float32)
    
    #negs = []

    for x in X :
        
        ll, _ = compute_log_likelihood(x, cond, sigma)

        ell = ell + ll
        
        #negs.append(s)


    return ell / N0 #, negs #averaging to get empirical entropy


def ensemble_log_likelihood_1D(X, cond) :
    """
    X :  N0 x C x H x W map
    cond : N x C x H x W map
    sigma : C x H x W map, dispersion of data on each variable-pixel point
    
    compute -log(p(X|cond)) (as C x H x W map) with the hypothesis of a sum of of a sum of 
    Gaussian conditional distributions with same spread and independent elements
    X1, .., X_N0
    
    """

    N0, C = X.shape

    sigma = np.std(cond, axis=0) / sqrt(N0)

    ell = np.zeros((C,)).astype(np.float32)

    for x in X :

        ell = ell + compute_log_likelihood(x, cond, sigma)

    return ell / N0 #averaging for fun (keeping cardinal-independent quantities)



def ensemble_entropy(X, cond) :
    """
    
    given a sample distribution of shape N x C x H x W
    compute the entropy for each degree of freedom
    
    """

    N0, C, H, W = X.shape

    sigma = np.std(cond, axis=0) / sqrt(N0)

    entropy = np.zeros((C, H, W)).astype(np.float32)

    for x in X :

        pX = (np.exp(-((x - cond)**2) / (2 * sigma**2)) / (np.sqrt(2 * pi) * sigma)).mean(axis=0)

        ell = compute_log_likelihood(x, cond, sigma)

        entropy =  entropy + pX * ell

    return entropy


def ensemble_entropy_1D(X, cond) :

    N0, C = X.shape

    sigma = np.std(cond, axis=0) / sqrt(N0 / 2.0)

    entropy = np.zeros((C,)).astype(np.float32)

    for x in X :

        pX = (np.exp(-((x - cond)**2) / (2 * sigma**2)) / (sqrt(2 * pi)* sigma)).mean()

        ell = compute_log_likelihood(x, cond, sigma)

        entropy =  entropy + pX * ell

    return entropy


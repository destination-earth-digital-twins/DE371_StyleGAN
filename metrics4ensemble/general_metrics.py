#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:42:42 2022

@author: brochetc


General metrics


"""


import numpy as np


############################ General simple metrics ###########################

def simple_variance(X) : 
    """
    X :  N x C H x W array
    
    Returns variance of X along the first dimension
    """
    
    return X.var(axis = 0)

def variance_diff(X, cond):
    """
    Maps of variance difference between ensemble X and condition cond
    
    X :  N x C x H x W array
    
    cond  : N_c x C x H x W array
    
    Returns :
        
        Diff : C x H x W array
    """
    
    Diff =  (X.var(axis=0) - cond.var(axis = 0))
    
    return Diff

def std_diff(X, cond):
    """
    Maps of variance difference between ensemble X and condition cond
    
    X :  N x C x H x W array
    
    cond  : N_c x C x H x W array
    
    Returns :
        
        Diff : C x H x W array
    """
    
    Diff =  (X.std(axis=0) - cond.std(axis = 0))
    
    return Diff
    
def relative_std_diff(X, cond) :
    """

    Maps of variance difference between ensemble X and condition cond
    
    X :  N x C x H x W array
    
    cond  : N_c x C x H x W array
    
    Returns :
        
        ratio : C x H x W array

    """
    
    ratio =  (X.std(axis=0) - cond.std(axis=0)) / (X.std(axis =0))
    
    return ratio

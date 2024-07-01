#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:49:12 2023

@author: brochetc

Multiple losses experiment launching

"""

import subprocess
from datetime import date
import pickle

#lam_MSE = [1.0,0.8,0.6,0.4,0.2,0.0]

lam_MSE = [1.0]
script_dir = '/home/mrmn/brochetc/restyle-encoder/scripts/'


states = {'l2' : [], 'lpips' : [], 'lscat' : [], 'lmoco' : [], 'lwnorm' : []}

for l2 in lam_MSE :
    
    for i in [1,2] :
        
        for j in [1,2]:
        
            lam_PIPS = 10**(float(i)) * l2
            
            lam_SCAT = 10**(float(j)) * l2
            
            lam_MOCO = 0
            
            lam_WNORM = 0
            
            states['l2'].append(l2)
            states['lpips'].append(lam_PIPS)
            states['lscat'].append(lam_SCAT)
            states['lmoco'].append(lam_MOCO)
            states['lwnorm'].append(lam_WNORM)
            
            subprocess.run(['bash', script_dir + 'multiple_launch_restyle.sh', str(l2), str(lam_PIPS), str(lam_SCAT), str(lam_MOCO), str(lam_WNORM)])

states['Date'] = date.today()
pickle.dump(open(script_dir + 'loss_training_{}.p'.format(date.today()), 'wb'), states)

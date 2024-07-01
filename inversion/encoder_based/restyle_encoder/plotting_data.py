#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:04:37 2023

@author: brochetc
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_dir = '/home/brochetc/priam-scratch/psp4arome_expe/'


me_train0 = pd.read_csv(data_dir + 'lr_0.001_l2_lambda_1.0_8iter/Instance_2/logs/metrics_train.csv')
me_test0 = pd.read_csv(data_dir + 'lr_0.001_l2_lambda_1.0_8iter/Instance_2/logs/metrics_test.csv')

me_train1 = pd.read_csv(data_dir + 'logs/metrics_train.csv')
me_test1 = pd.read_csv(data_dir + 'logs/metrics_test.csv')

me_train0 = me_train0.sort_values('Step')
me_train1 = me_train1.sort_values('Step')


iters0 = me_train0['Step']
iters1 = me_train1['Step']


smooth0 = [me_train0['loss_l2'][0]]

train_l2 = me_train1['loss_l2']
smooth1 = [me_train1['loss_l2'][0]]

for i in range(len(me_train0)-1) :
    smooth0.append(0.9 * smooth0[-1] + 0.1 *  me_train0['loss_l2'][i])

for j in range(len(me_train1)-1):
    smooth1.append(0.9 * smooth1[-1] + 0.1 *  me_train1['loss_l2'][j])

test_l2 = me_test0['loss_l2']

test_l21 = me_test1['loss_l2']


iters_test0 = me_test0['Step']
iters_test1 = me_test1['Step']



plt.plot(iters0, smooth0, 'b--',linewidth = 2)
plt.plot(iters_test0, test_l2, 'bo-',linewidth = 4, label = '8 refinements')

plt.plot(iters1, list(train_l2[:70]) + smooth1[70:], 'r--',linewidth = 2)
plt.plot(iters_test1, test_l21, 'ro-',linewidth = 4, label = '5 refinements')

#plt.plot(iters0, [1e-4 for i in range(len(iters0))], 'k--',linewidth = 2, label = 'Optimisation baseline' )

plt.yscale('log')
plt.ylim(2.5e-4,5e-3)
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()
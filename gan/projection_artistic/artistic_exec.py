#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:52:42 2022

@author: brochetc
"""
import argparse
import artistic as art
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--Path_samples', type = str,  default ='')
    parser.add_argument('--Path_out', type = str,  default ='')
    parser.add_argument('--data_dir', type = str, default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    params = parser.parse_args()

    CI = (78, 206, 55, 183)

    Maxs = np.load(params.data_dir+'MaxNew_4_var.npy')[1:4].reshape(3, 1, 1)
    Means = np.load(params.data_dir+'Mean_4_var.npy')[1:4].reshape(3, 1, 1)

    var_names = [('u', 'm/s'), ('v', 'm/s'), ('t2m', 'K')]
    index = 0
    lat_dim = 64

    Ens_proj_var_wplus = np.load(params.Path_samples+f'Fsemble_{lat_dim}_3.0_3.0.npy')
    Ens_real_var = np.load(params.Path_samples+f'Rsemble_{lat_dim}_3.0_3.0.npy')

    n_plots = 1
    channels = 3
    data = np.zeros((n_plots, channels, 128, 128))

    data0 = art.standardize_samples(Ens_real_var, normalize=[0], norm_vectors=(Means, Maxs),
                                    chan_ind=[0, 1, 2], ref_chan_ind=[0, 1, 2])[index]

    data2 = art.standardize_samples(Ens_proj_var_wplus, normalize=[0], norm_vectors=(Means, Maxs),
                                    chan_ind=[0, 1, 2], ref_chan_ind=[0, 1, 2])[index]


    data[0] = abs(data2 - data0)  # data1-data0

    can = art.canvasHolder("SE_for_GAN", 128, 128)

    Datamax = data.max(axis=(0, 2, 3))

    Datamin = data.min(axis=(0, 2, 3))

    print("data shape is", data.shape)

    can.plot_data_normal(data, var_names, params.Path_out, f'artistic_{lat_dim}_index_{index}.jpg', contrast=True,
                         cvalues=(Datamin, Datamax))

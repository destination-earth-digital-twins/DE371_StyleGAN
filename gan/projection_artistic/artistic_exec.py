#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:52:42 2022

@author: brochetc
"""

import artistic as art
import numpy as np
import random
import matplotlib.pyplot as plt
from glob import glob

data_dir = '/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/IS_1_1.0_0_0_0_0_0_256_done_with_8_var/'
Path_samples = '/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/inversion/Database_latent/W_plus/'
Path_out = '/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/inversion/PostProcessing/'

#Path_noise = '/home/moldovang/Documents/postproc_GAN/noise_injection/'

#data_dir_vent='/home/brochetc/Bureau/These/presentations_these/Deuxieme annee/longueurs de correlation/data/data_uv_8_0.0001/'
#data_dir_temp = '/home/brochetc/Bureau/These/presentations_these/Deuxieme annee/longueurs de correlation/data/data_t2m_8_0.0005/'

CI = (78, 206, 55, 183)

Maxs = np.load(data_dir+'max_with_8_var.npy')[1:4].reshape(3, 1, 1)
Means = np.load(data_dir+'mean_with_8_var.npy')[1:4].reshape(3, 1, 1)


if __name__ == "__main__":

    var_names = [('u', 'm/s'), ('v', 'm/s'), ('t2m', 'K')]
    index = 0
    lat_dim = 64
    #var_names=[('v', 'm/s')]

    # data_f = [1016,274,721] # 274 150
    #data_f = [184, 771, 62]

    # data_f=[916,449,733]
    # data_f = [1016, 228, 721] #(8 3 58500)
    #data_f = random.sample(range(256),3)

    # print(data_f)

    #data_flist=[Data_fake[data_f[j]] for j in range(3)]
    #del Data_fake

    # data_r_name=random.sample(litrue,1)[0][len(data_dir):]
    #data_r_name = '_sample19117.npy'
    # print(data_r_name)
    Ens_proj_var_wplus = np.load(Path_samples+f'Fsemble_{lat_dim}_3.0_3.0.npy')
    #Ens_proj_var_w = np.load('Sample_fake_0_w.npy' )
    Ens_real_var = np.load(Path_samples+f'Rsemble_{lat_dim}_3.0_3.0.npy')

    Ens_proj_var_wplus = Ens_proj_var_wplus
    #Ens_proj_var_w = Ens_proj_var_w
    Ens_real_var = Ens_real_var

    n_plots = 1
    channels = 3  # only 1 variable plot
    data = np.zeros((n_plots, channels, 128, 128))

    # data0 = Data_real
    # data0 = data0[0,0,:,:]
    # data0 = np.expand_dims(data0, axis=0)
    # data0 = np.expand_dims(data0, axis=0)

    # Data_fake = Data_fake[0,0,:,:]
    # Data_fake = np.expand_dims(Data_fake, axis=0)
    # Data_fake = np.expand_dims(Data_fake, axis=0)

    # print(data0.shape)

    data0 = art.standardize_samples(Ens_real_var, normalize=[0], norm_vectors=(Means, Maxs),
                                    chan_ind=[0, 1, 2], ref_chan_ind=[0, 1, 2])[index]
    #data1 = art.standardize_samples(Ens_proj_var_w, normalize=[0], norm_vectors=(Means,Maxs), chan_ind =[0, 1, 2], ref_chan_ind =[0, 1, 2])
    data2 = art.standardize_samples(Ens_proj_var_wplus, normalize=[0], norm_vectors=(Means, Maxs),
                                    chan_ind=[0, 1, 2], ref_chan_ind=[0, 1, 2])[index]

    #data[0] = art.standardize_samples(Ens_real_var, normalize=[0], norm_vectors=(Means,Maxs), chan_ind =[0, 1, 2], ref_chan_ind =[0, 1, 2])
    #data[1] = art.standardize_samples(Ens_proj_var_w, normalize=[0], norm_vectors=(Means,Maxs), chan_ind =[0, 1, 2], ref_chan_ind =[0, 1, 2])
    #data[2] = art.standardize_samples(Ens_proj_var_wplus, normalize=[0], norm_vectors=(Means,Maxs), chan_ind =[0, 1, 2], ref_chan_ind =[0, 1, 2])

    data[0] = abs(data2 - data0)  # data1-data0
    #data[1] = abs(data2 - data0)

    #data[0] = art.standardize_samples(Data_noise_var,norm_vectors=(Means,Maxs), chan_ind =[0, 1, 2], ref_chan_ind =[0, 1, 2])
    #data[0] = art.standardize_samples(Data_noise_var_4,norm_vectors=(Means,Maxs), chan_ind =[0, 1, 2], ref_chan_ind =[0, 1, 2])
    #data[1] = art.standardize_samples(Data_noise_var_32,norm_vectors=(Means,Maxs), chan_ind =[0, 1, 2], ref_chan_ind =[0, 1, 2])
    can = art.canvasHolder("SE_for_GAN", 128, 128)
    # for j in range(n_plots):

    #data[j+1] = art.standardize_samples(Data_fake, normalize=[0],norm_vectors=(Means,Maxs), chan_ind =[0], ref_chan_ind =[0])
    #data[j] = art.standardize_samples(np.expand_dims(Data_interp[j], axis=0), normalize=[0],norm_vectors=(Means,Maxs), chan_ind =[0], ref_chan_ind =[0])
    # data_name="_sample1911.npy"

    Datamax = data.max(axis=(0, 2, 3))
    #Datamax[0] = Datamax[0]*0.99
    # Datamax[1]=Datamax[1]*0.85
    # Datamax[2]=Datamax[2]*0.99

    Datamin = data.min(axis=(0, 2, 3))

    print("data shape is", data.shape)

    can.plot_data_normal(data, var_names, Path_out, f'artistic_{lat_dim}_index_{index}.jpg', contrast=True,
                         cvalues=(Datamin, Datamax))
    #can.plot_data_wind(data[:,0:2,:,:], path_plot_full,'new_artistic_wind_63_357000_cbr.png',withQuiver=False)
    #can.plot_data_wind(data[:,0:2,:,:], path_plot_full,'new_artistic_wind_quiver_63_357000_cbr.png',withQuiver=True)

    # can.plot_data_normal(data,var_names,path_plot, 'new_artistic0_progan.png', contrast=True,
    #                     cvalues=(Datamin, Datamax))

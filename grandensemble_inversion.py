#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:21:37 2023

@author: brochetc

Main pod sampling script

"""
import torch
import argparse
from gan.model.stylegan2 import Generator
import os
import numpy as np
import perturbation.inversion as inv
from time import perf_counter
from collections import OrderedDict

import pandas as pd
from datetime import date, timedelta, datetime
import perturbation.utils as utils


torch.manual_seed(42) #reproducibility of runs

if __name__=="__main__" :


    
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################

    parser.add_argument('--ckpt_dir', type = str, 
                        default ='/scratch/work/brochetc/Exp_StyleGAN/Set_1/stylegan2_stylegan_dom_256_lat-dim_512_bs_4_0.002_0.002_ch-mul_2_vars_u_v_t2m_noise_True/Instance_14/models/000024.pt')
    parser.add_argument('--real_data_dir', type = str, 
                        default ='/scratch/work/brochetc/grandEnsemble/AROME/')
    parser.add_argument('--output_dir',type = str, 
                        default ='/scratch/work/brochetc/Exp_StyleGAN/Inversion_GE/')
    parser.add_argument("--pack_dir", type=str, default = '/scratch/work/brochetc/Exp_StyleGAN/Pack_GE/') # storing "packed" (normalized) real data
    
    parser.add_argument('--mean_file', type=str, default='Mean_4_var.npy')
    parser.add_argument('--max_file', type=str, default='MaxNew_4_var.npy')
    
    parser.add_argument('--device', type=str, default='cuda:0')
    ############################ INVERSION PARAMETERS #################    

    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float, 
        default=0.25,
        help="duration of the learning rate decay",
    )
    
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    
  
    parser.add_argument(
        "--noise", type=float, default=0.005, help="strength of the noise level"
    )
    
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    
    parser.add_argument("--invstep", type=int, default=1000, help="optimize iterations")
    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
    parser.add_argument("--Shape", type=tuple, default=(3,256,256), help='size of the samples')
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=10e5,
        help="weight of the noise regularization (inversion)",
    )

    parser.add_argument('--start_member', type=int, default=0)
    parser.add_argument('--stop_member', type=int, default=874)

    parser.add_argument('--loss', type=str, default='mse', choices = ['mse', 'mae'])
    parser.add_argument("--loss_intens", type=float, default=1.0, help="weight of the pixel loss")

    parser.add_argument("--inv_checkpoints", type=utils.str2intlist, default=[400,800,1000])


    ########################## CONTROL of Data to invert ######################

    parser.add_argument("--leadtimes", type=utils.str2intlist, default=[6,12,18,24,30,36,42])
    params = parser.parse_args()

    ################## loading normalisation data and deciding members slicing ##

    Means = np.load(f'{params.real_data_dir}{params.mean_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)

    Maxs = np.load(f'{params.real_data_dir}{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)

    list_members = range(params.start_member, params.stop_member, 16)

    ################ loading network #################

    device = params.device if torch.cuda.is_available() else 'cpu'

    G = Generator(params.Shape[1], 512,n_mlp=8,nb_var=params.Shape[0])


    #print('###########################################"##################################################################################################################')
    ckpt = torch.load(params.ckpt_dir, map_location='cpu')['g_ema']

    if 'module' in list(ckpt.items())[0][0]: #juglling with Pytorch versioning and different module packaging
        ckpt_adapt = OrderedDict()
        for k in ckpt.keys():
            k0 = k[7:]
            ckpt_adapt[k0] = ckpt[k]
        G.load_state_dict(ckpt_adapt)


    else:
        G.load_state_dict(ckpt_dic)
    G.eval()

    G = G.to(device)

    ################### producing latent mean #######

    if not os.path.exists(f'{params.output_dir}latent_mean.npy'):

        latent_z = torch.empty(10000, 512).normal_().to(device)
        with torch.no_grad():
            w = G.style(latent_z)

        latent_mean = w.mean(dim=0).detach().cpu()

        np.save(f'{params.output_dir}latent_mean.npy',latent_mean.numpy())
    else : 

        lm = np.load(f'{params.output_dir}latent_mean.npy').astype(np.float32)
        latent_mean = torch.tensor(lm, dtype = torch.float32)


    #################### main loop ##################

    for j,stmb in enumerate(list_members):

        start = stmb 
        if j<len(list_members)-1:
            stop = list_members[j+1] - 1
        else :
            stop = params.stop_member
        
        for lt in params.leadtimes:
            print(start, stop, lt)
            if not os.path.exists(params.output_dir +f'w_{start}_{stop}_{lt}_1000.npy'): #checking for already teer

                Ens_r = utils.collate_ensemble(params.real_data_dir, start, stop, lt, params.var_indices)

                Ens_r = torch.tensor(0.95 * (Ens_r - Means) / Maxs, dtype = torch.float32)

                np.save(params.pack_dir+f'Rsemble_{start}_{stop}_{lt}.npy', Ens_r.numpy().astype(np.float32))

                params.date_index = f'{start}_{stop}'
                params.lt_index = lt
            
                inv.optimize(Ens_r, G, latent_mean, device, params)

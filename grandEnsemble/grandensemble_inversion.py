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
import inversion.optimization_based.inversion as inv
from collections import OrderedDict
import pandas as pd
from datetime import date, timedelta, datetime
import utils.utils as utils


torch.manual_seed(42) #reproducibility of runs

if __name__=="__main__" :


    
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################

    parser.add_argument('--ckpt_dir', type = str, default ='')
    parser.add_argument('--real_data_dir', type = str, default ='')
    parser.add_argument('--output_dir',type = str, default ='')
    parser.add_argument("--pack_dir", type=str, default = '') # storing "packed" (normalized) real data
    
    # Dataset information
    parser.add_argument("--normalization", type=str, default="meanmax", choices=["minmax", "meanmax"])
    parser.add_argument('--max_file', type=str, default='MaxNew_4_var.npy') # use 'MaxNew_4_var.npy' if AROME data # max_rr_log.npy
    parser.add_argument('--mean_file', type=str, default='Mean_4_var.npy') # not used if minmax normalization
    parser.add_argument('--min_file', type=str, default='min_rr_log.npy')  # not used if meanmax normalization
    
    parser.add_argument('--device', type=str, default='cuda')

    ############################ SEQUENCE PARAMETERS #################    
    parser.add_argument('--multi_timestep_mode', action='store_true')
    parser.add_argument('--nb_timesteps', type=int, default=15)
    parser.add_argument('--timestep_period', type=int, default=3)
    parser.add_argument('--stack_sample_along_time_and_variable', action='store_true')
    parser.add_argument('--g_channels', type=int, default=3)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    
    ############################ INVERSION PARAMETERS #################    

    parser.add_argument("--lr_rampup",type=float,default=0.05,help="duration of the learning rate warmup")
    parser.add_argument("--lr_rampdown",type=float, default=0.25,help="duration of the learning rate decay")
    
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    
    parser.add_argument("--noise_strength", type=float, default=0.005, help="strength of the noise level")
    parser.add_argument("--noise_ramp",type=float,default=0.75,help="duration of the noise level decay")
    
    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
    parser.add_argument("--Shape", type=tuple, default=(3,256,256), help='size of the samples')
    parser.add_argument("--crop_indices", type=int, nargs='+', default=[0,256,0,256])

    # Noise optimization and loss noise parameter
    parser.add_argument("--noise_optimize", action='store_true', help="joint optimization of noise and latent code (1) or latent code optimization only (0)?")
    parser.add_argument("--feature_optimize", action='store_true', help="to enable optimization of feature map")
    parser.add_argument("--feature_id", type=int, default=6, help="features to optimize")
    parser.add_argument("--lambda_features", type=float, default=1, help="weight of the noise regularization")

    parser.add_argument("--lambda_noise", type=float, default=1e5, help="weight of the noise regularization")
    # In case noise_optimize=0, the lambda_noise is not taken into account in the loss computation
    parser.add_argument("--fixed_noise", action='store_true', help="Fixing the noise during optimization")

    # Parameter related to pixel loss 
    parser.add_argument('--pixel_loss_type', type=str, default='mse', choices = ['mse', 'mae'])
    parser.add_argument("--lambda_pixel", type=float, default=10.0, help="weight of the (mae/mse) pixel loss")
    
    # Focal Frequency Loss
    parser.add_argument("--lambda_focal_frequency_loss", type=float, default=0.0, help="weight of the vgg (perceptual) loss")

    # Perceptual Loss / LPIPS loss
    parser.add_argument("--lambda_lpips_loss", type=float, default=1.0, help="weight of the LPIPS loss")
    parser.add_argument("--lambda_perceptual_loss", type=float, default=1.0, help="weight of the vgg (perceptual) loss")
    parser.add_argument("--resize_input", type=float, default=0.0, help="resize input for vgg loss")
    parser.add_argument("--network_type", type=str, default='vgg16', choices=['vgg16','vgg11','vgg13','vgg19','alexnet','squeezenet1_1','resnet18','resnet34','resnet50','resnet101','resnet152','set_vit_b_16'])
    parser.add_argument("--pre_trained", action='store_true')
    parser.add_argument("--features_after_relu", action='store_true')
    parser.add_argument("--channel_computation", type=str, default='sol2', choices = ['sol1', 'sol2', 'sol3', 'sol4', 'sol5'], 
                    help="Either we compute layer by layer and member per member but we have to triple th einput to make it rgb or all in one (naive)")
    parser.add_argument("--network_dir", type=str, default='/project/home/p200177/DE_371/resources/network_for_perceptual_loss/', help="Insert a path")
    parser.add_argument("--style_layers", type=utils.str2intlist, default=[], help="style layers to include in vgg loss computation")
    parser.add_argument("--feature_layers", type=utils.str2intlist, default=[0,1,2,3], help="feature layers to include in vgg computation")
    parser.add_argument("--alpha_feature", type=float, default=1.0, help="weight of the feature/content loss")
    parser.add_argument("--alpha_style", type=float, default=0.01, help="weight of the style loss")
    parser.add_argument("--multi_scale_perceptual_loss",  action='store_true')
    
    parser.add_argument("--invstep", type=int, default=1000, help="optimize iterations")
    parser.add_argument("--inv_checkpoints", type=utils.str2intlist, default=[100,200,300,400,500,1000])
    parser.add_argument("--plot_checkpoint", action='store_true')
    parser.add_argument("--plot_gif", action='store_true', help="Plotting gif of optimization")
    
    # lambda_ms_ssim
    parser.add_argument("--lambda_ms_ssim", type=float, default=0, help="weight of the MS-SSIM loss")
    
    parser.add_argument("--inv_checkpoints", type=utils.str2intlist, default=[500,1000,1500,2000])
    parser.add_argument("--plot_checkpoint", action='store_true')

    parser.add_argument("--seed", type=int, default=42)

    ########################## CONTROL of Data to invert ######################
    parser.add_argument('--start_member', type=int, default=0)
    parser.add_argument('--stop_member', type=int, default=874)
    parser.add_argument("--leadtimes", type=utils.str2intlist, default=[6,12,18,24,30,36,42])
    params = parser.parse_args()

    # create output and pack directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    if not os.path.exists(params.pack_dir):
        os.makedirs(params.pack_dir)

    # set the seed for reproduciibility of runs
    torch.manual_seed(params.seed)

    ################## loading normalisation data and deciding members slicing ##
    Means = np.load(f'{params.real_data_dir}{params.mean_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    Maxs = np.load(f'{params.real_data_dir}{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    list_members = range(params.start_member, params.stop_member, 16)

    ################ loading network #################
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
        G.load_state_dict(ckpt)
    G.eval()
    G = G.to(params.device)

    ################### producing latent mean #######
    if not os.path.exists(f'{params.output_dir}latent_mean.npy'):
        latent_z = torch.empty(10000, 512).normal_().to(params.device)
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
            if not os.path.exists(params.output_dir +f'w_{start}_{stop}_{lt}_2000.npy'): #checking for already teer

                Ens_r = utils.collate_ensemble(params.real_data_dir, start, stop, lt, params.var_indices)
                Ens_r = torch.tensor(0.95 * (Ens_r - Means) / Maxs, dtype = torch.float32)
                np.save(params.pack_dir+f'Rsemble_{start}_{stop}_{lt}.npy', Ens_r.numpy().astype(np.float32))
                params.date_index = f'{start}_{stop}'
                params.lt_index = lt
            
                inv.optimize(
                    Ens_r=Ens_r,
                    g_ema=G,
                    latent_mean=latent_mean,
                    device=params.device,
                    params=params
                )

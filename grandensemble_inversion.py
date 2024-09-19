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
                        default ='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt')
    parser.add_argument('--real_data_dir', type = str, default ='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/grandEnsemble/AROME/')
    parser.add_argument('--output_dir',type = str, default ='/project/scratch/p200177/DE_371/victorsanchez/results/Grand_Ensemble/Inversion')
    parser.add_argument("--pack_dir", type=str, default = '/project/scratch/p200177/DE_371/victorsanchez/results/Grand_Ensemble/Pack/') # storing "packed" (normalized) real data
    
     # Dataset information
    parser.add_argument('--mean_file', type=str, default='Mean_4_var.npy')
    parser.add_argument('--max_file', type=str, default='MaxNew_4_var.npy')
    
    parser.add_argument('--device', type=str, default='cuda')
    ############################ INVERSION PARAMETERS #################    

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
    parser.add_argument("--lambda_noise", type=float, default=1e5, help="weight of the noise regularization")
    # In case noise_optimize=0, the lambda_noise is not taken into account in the loss computation
    parser.add_argument("--fixed_noise", action='store_true', help="Fixing the noise during optimization")

    # Parameter related to pixel loss 
    parser.add_argument('--pixel_loss_type', type=str, default='mse', choices = ['mse', 'mae'])
    parser.add_argument("--lambda_pixel", type=float, default=10.0, help="weight of the (mae/mse) pixel loss")
    
    # Parameter related to perceptual loss 
    parser.add_argument("--optimize_features_computation", action='store_true', help="Compute the features of original ensemble only once")
    parser.add_argument("--progressive_loss_mode", action='store_true', help="Progressive Loss between pixel loss and perceptual loss | Start : Only MSE | End : Only Perceptual")


    # LPIPS
    parser.add_argument("--lpips_pnet", type=str, default='alex', choices=['alex','vgg','squeeze'], help="network type for lpips loss")
    parser.add_argument("--lpips_pnet_tune", action='store_true', help="tuning the weights of the pnet")
    parser.add_argument("--lpips_pnet_state_dict_path", type=str, default='/home/users/u101833/project/DE371_StyleGAN/inversion/PerceptualSimilarity/lpips/weights_pnets/alex_random.pth', help="path to lpips pre-trained network weights")
    parser.add_argument("--lambda_lpips", type=float, default=0.0, help="weight of the lpips (perceptual) loss")

    parser.add_argument("--lpips_mode", action='store_true', help="if lpips mode=False, it act like simple vgg")
    parser.add_argument("--lpips_linear_layers_state_dict_path", type=str, default='/home/users/u101833/project/DE371_StyleGAN/inversion/PerceptualSimilarity/lpips/weights_linear_layers/v0.1/vgg.pth', help="path to liunear layer lpips")
    
    # VGG
    parser.add_argument("--lambda_vgg", type=float, default=1.0, help="weight of the vgg (perceptual) loss")
    parser.add_argument("--vgg_computation", type=str, default='sol2', choices = ['sol1', 'sol2', 'sol3', 'sol4', 'sol5'], 
                        help="Either we compute layer by layer and member per member but we have to triple th einput to make it rgb or all in one (naive)")
    parser.add_argument("--vgg_state_dict_path", type=str, default='/project/scratch/p200177/DE_371/resources/vgg_weights/vgg16-random.pth', help="Insert a path")
    parser.add_argument("--vgg_style_layers", type=utils.str2intlist, default=[], help="style layers to include in vgg loss computation")
    parser.add_argument("--vgg_feature_layers", type=utils.str2intlist, default=[0,1,2,3], help="feature layers to include in vgg computation")
    parser.add_argument("--vgg_alpha_feature", type=float, default=1.0, help="weight of the feature/content loss")
    parser.add_argument("--vgg_alpha_style", type=float, default=0.01, help="weight of the style loss")
    parser.add_argument("--vgg_loss_after_step", type=float, default=0, help="compute the vgg loss only after a given number of steps")

    # lambda_ms_ssim
    parser.add_argument("--lambda_ms_ssim", type=float, default=0, help="weight of the MS-SSIM loss")

    parser.add_argument("--invstep", type=int, default=2000, help="optimize iterations")
    
    
    parser.add_argument('--start_member', type=int, default=0)
    parser.add_argument('--stop_member', type=int, default=874)

    
    parser.add_argument("--inv_checkpoints", type=utils.str2intlist, default=[500,1000,1500,2000])
    parser.add_argument("--plot_checkpoint", action='store_true')

    parser.add_argument("--seed", type=int, default=42)

    ########################## CONTROL of Data to invert ######################

    parser.add_argument("--leadtimes", type=utils.str2intlist, default=[6,12,18,24,30,36,42])
    params = parser.parse_args()

    # create output and pack directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    if not os.path.exists(params.pack_dir):
        os.makedirs(params.pack_dir)

    # set the seed for reproduciibility of runs
    seed = params.seed
    torch.manual_seed(seed)

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
        G.load_state_dict(ckpt_dic)
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
            
                inv.optimize(Ens_r, G, latent_mean, params.device, params)

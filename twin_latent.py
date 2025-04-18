#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs ensemble forecast inversion using a pre-trained StyleGAN2 model.

The code uses command-line arguments for setting directories, inversion parameters, and data control parameters.
The inversion is performed for a specified set of dates and lead times, generating latent code representations for real-ensemble data and saving the results.

Please make sure to configure the directory paths, parameters, and other settings based on your specific environment before running the script.

"""
import torch
import argparse
import os
import numpy as np
import yaml
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from encoders.models.psp import pSp
from inversion.hybrid_based.inversion import init_latent_restyle, init_latent_psp_e4e, init_latent_featureStyle, init_latent_inDomain
from inversion.encoder_based.inversion import inversion_restyle, inversion_psp_e4e, inversion_featureStyle, inversion_inDomain
from inversion.encoder_based.encoder_utils import log_images_diff
from encoders.models.e4e import e4e
from encoders.models.in_domain import inDomain
from encoders.models.feature_style_encoder.feature_style_module import FeatureStyleModule
import inversion.optimization_based.inversion as inv
from gan.model.stylegan2 import Generator
from collections import OrderedDict
import inversion.optimization_based.inversion as inv
import utils.utils as utils
from ast import literal_eval as make_tuple
torch.manual_seed(42) #reproducibility of runs

if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--inversion_type', default='optimization', type=str, choices=["optimization","encoder","hybrid","optimization_amse_vgg"], help='Type of inversion')

    ########################### Encoder-related parameters ###########################
    parser.add_argument('--encoder_framework_type', default='FeatureStyle', type=str, choices=["pSp", "e4e", "restyle-pSp", "restyle-e4e", "FeatureStyle", "inDomain"], help='Type of encoder')
    parser.add_argument('--encoder_checkpoint_dir', default ='', type=str, help='Path to ReStyle model checkpoint')
    parser.add_argument('--dataset_type', default='arome_encode', type=str, help='Type of dataset/experiment to run')
    parser.add_argument('--encoder_type', default='ResNetBackboneEncoder', type=str, help='Which encoder to use')
    parser.add_argument('--input_nc', default=6, type=int, help='Number of input image channels to the ReStyle encoder. Should be set to 6.')
    parser.add_argument('--output_size', default=256, type=int, help='Output size of generator')
    parser.add_argument('--n_vars', default=3, type=int, help='Number of variables as channels')
    parser.add_argument("--plot_checkpoint", action='store_true')
    parser.add_argument('--plot_gif', action='store_true')
    parser.add_argument('--plot_loss_evolution', action='store_true')

    parser.add_argument("--train_discriminator", action='store_true')

    # arguments for iterative encoding
    parser.add_argument('--n_iters_per_batch', default=10, type=int,help='Number of forward passes per batch during training')
    parser.add_argument('--n_iters_per_batch_checkpoint', type=utils.str2intlist, default=[1,5,10], help='Number of forward passes per batch during training')
    
    ########################### Directories ###########################

    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str,default='')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, default='')
    # Pack Directory - PATH where the packed ensembles will be saved

    parser.add_argument("--pack_dir", type=str, default = '') # storing "packed" (normalized) real data
    parser.add_argument('--ckpt_dir', type = str, default ='')
 

    # Dataset information
    parser.add_argument("--normalization", type=str, default="minmax", choices=["minmax", "meanmax"])
    parser.add_argument('--max_file', type=str, default='max_rr_log.npy') # use 'MaxNew_4_var.npy' if AROME data # max_rr_log.npy
    parser.add_argument('--mean_file', type=str, default='mean_rr_log.npy') # not used if minmax normalization
    parser.add_argument('--min_file', type=str, default='min_rr_log.npy')  # not used if meanmax normalization
    parser.add_argument('--save_normalized_sample', action='store_true')

    parser.add_argument('--device', type=str, default='cuda')

    ############################ SEQUENCE PARAMETERS #################    
    parser.add_argument('--multi_timestep_mode', action='store_true')
    parser.add_argument('--nb_timesteps', type=int, default=15)
    parser.add_argument('--timestep_period', type=int, default=3)
    parser.add_argument('--stack_sample_along_time_and_variable', action='store_true')
    parser.add_argument('--g_channels', type=int, default=4)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    
    
    ############################ INVERSION PARAMETERS #################    
    parser.add_argument("--lr_rampup",type=float,default=0.05,help="duration of the learning rate warmup")
    parser.add_argument("--lr_rampdown",type=float, default=0.25,help="duration of the learning rate decay")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    
    parser.add_argument("--noise_strength", type=float, default=0.005, help="strength of the noise level")
    parser.add_argument("--noise_ramp",type=float,default=0.75,help="duration of the noise level decay")
    parser.add_argument("--feature_optimize", action='store_true', help="to enable optimization of feature map")
    parser.add_argument("--feature_id", type=int, default=6, help="features to optimize")
    parser.add_argument("--feature_scale", type=float, default=1, help="features scale when inserting")
    parser.add_argument("--lambda_features", type=float, default=1, help="weight of the noise regularization")

    # Noise optimization and loss noise parameter
    parser.add_argument("--noise_optimize", action='store_true', help=" joint optimization of noise and latent code (1) or latent code optimization only (0)?")
    parser.add_argument("--lambda_noise", type=float, default=1e5, help="weight of the noise regularization")
    # In case noise_optimize=0, the lambda_noise is not taken into account in the loss computation
    parser.add_argument("--fixed_noise", action='store_true', help="Fixing the noise during optimization")

    # Parameter related to pixel loss 
    parser.add_argument('--pixel_loss_type', type=str, default='amse', choices = ['mse', 'mae','amse','wamse','wmse'])
    parser.add_argument("--lambda_pixel", type=float, default=10.0, help="weight of the (mae/mse) pixel loss")
    
        
    # Focal Frequency Loss
    parser.add_argument("--lambda_focal_frequency_loss", type=float, default=0.0, help="weight of the vgg (perceptual) loss")

    # VGG
    parser.add_argument("--lambda_lpips_loss", type=float, default=0.0, help="weight of the LPIPS loss")
    parser.add_argument("--lambda_perceptual_loss", type=float, default=1.0, help="weight of the vgg (perceptual) loss")
    parser.add_argument("--resize_input", type=float, default=0.0, help="resize input for vgg loss")
    parser.add_argument("--network_type", type=str, default='vgg16', choices=['vgg16','vgg11','vgg13','vgg19','alexnet','squeezenet1_1','resnet18','resnet34','resnet50','resnet101','resnet152','set_vit_b_16'])
    parser.add_argument("--pre_trained", action='store_true')
    parser.add_argument("--features_after_relu", action='store_true')
    parser.add_argument("--channel_computation", type=str, default='sol2', choices = ['sol1', 'sol2', 'sol3', 'sol4', 'sol5'], 
                    help="Either we compute layer by layer and member per member but we have to triple th einput to make it rgb or all in one (naive)")
    parser.add_argument("--network_dir", type=str, default='', help="Insert a path")
    parser.add_argument("--style_layers", type=utils.str2intlist, default=[], help="style layers to include in vgg loss computation")
    parser.add_argument("--feature_layers", type=utils.str2intlist, default=[0,1,2,3], help="feature layers to include in vgg computation")
    parser.add_argument("--alpha_feature", type=float, default=1.0, help="weight of the feature/content loss")
    parser.add_argument("--alpha_style", type=float, default=0.01, help="weight of the style loss")
    parser.add_argument("--split_factor", type=int, default=2, help="splitting factor for patching")
    parser.add_argument("--multi_scale_perceptual_loss",  action='store_true')

    parser.add_argument("--invstep", type=int, default=2000, help="optimize iterations (default is 50 when hybrid-based and 1000 when optimization-based)")
    parser.add_argument("--inv_checkpoints", type=utils.str2intlist, default=[500,1000, 1500,2000])
    
    # lambda_ms_ssim
    parser.add_argument("--lambda_ms_ssim", type=float, default=0, help="weight of the MS-SSIM loss")
    
    parser.add_argument("--var_indices", type=utils.str2intlist, default=[0,1,2,3])
    parser.add_argument("--Shape", type=make_tuple, default=(4,256,256), help='size of the samples')
    parser.add_argument("--crop_indices", type=int, nargs='+', default=[0,256,0,256])

    ########################## CONTROL of Data to invert ######################
    parser.add_argument("--dates_file", type=str, help='csv file')
    parser.add_argument("--date_start", type=str, default = "2020-07-01")
    parser.add_argument("--date_stop", type=str, default = "2021-07-02")
    parser.add_argument("--leadtimes", type=utils.str2intlist, default=[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45])
    
    parser.add_argument("--seed", type=int, default=42)
    
    params = parser.parse_args()


    # fix some of the inputs
    params.Shape = tuple(params.Shape)
    params.crop_indices = tuple(params.crop_indices)

    # create output and pack directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    if not os.path.exists(params.pack_dir) and params.pack_dir != '':
        os.makedirs(params.pack_dir)

    # set the seed for reproduciibility of runs
    seed = params.seed
    torch.manual_seed(seed)

    ################## loading dates and file names ##
    df = pd.read_csv(params.real_data_dir + params.dates_file)
    df_date = df.copy()
    df_date['Date'] = pd.to_datetime(df_date['Date'])
    df_extract = df_date[(df_date['Date']>=params.date_start) & (df_date['Date']<=params.date_stop)]

    list_dates = df_extract['Date'].unique()



    #################### main loop ##################
    for date_ in list_dates:
        print(date_)
        datename = date_.strftime('%Y-%m-%d')
        print("\n===========================")
        for lt in params.leadtimes:
            params.date_index = datename
            params.lt_index = lt
            
            path_amse =  '/project/scratch/p200177/DE_371/angeliquebonamy/results/dates/GOOD_DATA/amse_vgg/inv_norm/amse/'+'invertFsemble_{}_{}_{}.npy'.format(params.date_index,lt, 2000)
            path_vgg =  '/project/scratch/p200177/DE_371/angeliquebonamy/results/dates/GOOD_DATA/amse_vgg/inv_norm/vgg/'+'invertFsemble_{}_{}_{}.npy'.format(params.date_index,lt, 2000)

            if  os.path.isfile(path_amse) and os.path.isfile(path_vgg):
                    
                data2plotamse = np.load(path_amse).astype(np.float32)[:,np.newaxis,:,:]
                data2plotvgg = np.load(path_vgg).astype(np.float32)[:,np.newaxis,:,:]
                
                empty_df = np.zeros(data2plotamse.shape,dtype=np.float32)
                empty_df[:,:,0,:,:]=data2plotamse[:,:,0,:,:]
                empty_df[:,:,1:4,:,:]=data2plotvgg[:,:,1:4,:,:]
                np.save(params.output_dir+'amse_vgg/invertFsemble_{}_{}_{}.npy'.format(params.date_index,params.lt_index,2000),np.squeeze(empty_df,axis=1))
            else:
                print("path does not exist",os.path.isfile(path_vgg),os.path.isfile(path_amse),path_amse,path_vgg)
                continue

            
            
            

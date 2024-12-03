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
from restyle_encoder.utils import common, train_utils
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import perturbation.utils as utils
from restyle_encoder.models.psp import pSp
from inversion.hybrid_based.inversion import init_latent_restyle, init_latent_psp_e4e, init_latent_featureStyle, init_latent_inDomain
from restyle_encoder.models.e4e import e4e
from restyle_encoder.models.in_domain import inDomain
from restyle_encoder.models.feature_style_encoder.feature_style_module import FeatureStyleModule
from generate_sample import humanbytes
from time import time
from inversion.encoder_based.utils import log_images_diff
import inversion.optimization_based.inversion as inv
from time import time

torch.manual_seed(42) #reproducibility of runs

if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################
    parser.add_argument('--encoder_framework_type', default='pSp', type=str, choices=["pSp", "e4e", "restyle-pSp", "restyle-e4e", "FeatureStyle", "inDomain"], help='Type of encoder')
    parser.add_argument('--checkpoint_path', default='/project/scratch/p200177/DE_371/victorsanchez/results/encoder/restyle_pSp_training/lr_0.001_vgg_lambda_1.0_resnet34=trained_10_iter/Instance_1/checkpoints/iteration_50000.pt', type=str, help='Path to ReStyle model checkpoint')
    parser.add_argument('--dataset_type', default='arome_encode', type=str, help='Type of dataset/experiment to run')
    parser.add_argument('--encoder_type', default='ResNetBackboneEncoder', type=str, help='Which encoder to use')
    parser.add_argument('--input_nc', default=6, type=int, help='Number of input image channels to the ReStyle encoder. Should be set to 6.')
    parser.add_argument('--output_size', default=256, type=int, help='Output size of generator')
    parser.add_argument('--n_vars', default=3, type=int, help='Number of variables as channels')
    parser.add_argument("--plot_checkpoint", action='store_true')

    parser.add_argument("--train_discriminator", action='store_true')

    # arguments for iterative encoding
    parser.add_argument('--n_iters_per_batch', default=10, type=int,help='Number of forward passes per batch during training')
    parser.add_argument('--n_iters_per_batch_checkpoint', type=utils.str2intlist, default=[1,5,10], help='Number of forward passes per batch during training')

    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str, default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, default ='/project/scratch/p200177/DE_371/victorsanchez/results/member_inversion/hybrid_inversion/test/inversion/')
    # Pack Directory - PATH where the packed ensembles will be saved
    parser.add_argument("--pack_dir", type=str, default = '/project/scratch/p200177/DE_371/victorsanchez/results/member_inversion/hybrid_inversion/test/pack/') # storing "packed" (normalized) real data
    
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

    # Progressive loss mode
    # action='store_true': 
    #   Sets the value to True if the argument is called without any value (e.g. --progressive_loss_mode)
    #   If the arguments is omitted, parser sets the value to False
    parser.add_argument("--progressive_loss_mode", action='store_true', help="Progressive Loss between pixel loss and perceptual loss | Start : Only MSE | End : Only Perceptual")

    # Noise optimization and loss noise parameter
    parser.add_argument("--noise_optimize", action='store_true', help="joint optimization of noise and latent code (1) or latent code optimization only (0)?")
    parser.add_argument("--lambda_noise", type=float, default=1e5, help="weight of the noise regularization")
    # In case noise_optimize=0, the lambda_noise is not taken into account in the loss computation
    parser.add_argument("--fixed_noise", action='store_true', help="Fixing the noise during optimization")

    # Parameter related to pixel loss 
    parser.add_argument('--pixel_loss_type', type=str, default='mse', choices = ['mse', 'mae'])
    parser.add_argument("--lambda_pixel", type=float, default=0.0, help="weight of the (mae/mse) pixel loss")
    
        
    # Focal Frequency Loss
    parser.add_argument("--lambda_focal_frequency_loss", type=float, default=0.0, help="weight of the vgg (perceptual) loss")

    # VGG
    parser.add_argument("--lambda_perceptual_loss", type=float, default=1.0, help="weight of the vgg (perceptual) loss")
    parser.add_argument("--resize_input", type=float, default=0.0, help="resize input for vgg loss")
    parser.add_argument("--network_type", type=str, default='vgg16', choices=['vgg16','vgg11','vgg13','vgg19','alexnet','squeezenet1_1','resnet18','resnet34','resnet50','resnet101','resnet152','set_vit_b_16'])
    parser.add_argument("--pre_trained", action='store_true')
    parser.add_argument("--features_after_relu", action='store_true')
    parser.add_argument("--channel_computation", type=str, default='sol2', choices = ['sol1', 'sol2', 'sol3', 'sol4', 'sol5'], 
                    help="Either we compute layer by layer and member per member but we have to triple th einput to make it rgb or all in one (naive)")
    parser.add_argument("--network_dir", type=str, default='/project/scratch/p200177/DE_371/resources/network_for_perceptual_loss/', help="Insert a path")
    parser.add_argument("--style_layers", type=utils.str2intlist, default=[], help="style layers to include in vgg loss computation")
    parser.add_argument("--feature_layers", type=utils.str2intlist, default=[0,1,2,3], help="feature layers to include in vgg computation")
    parser.add_argument("--alpha_feature", type=float, default=1.0, help="weight of the feature/content loss")
    parser.add_argument("--alpha_style", type=float, default=0.01, help="weight of the style loss")
    parser.add_argument("--split_factor", type=int, default=2, help="splitting factor for patching")
    parser.add_argument("--multi_scale_perceptual_loss",  action='store_true')

    parser.add_argument("--invstep", type=int, default=50, help="optimize iterations")
    parser.add_argument("--inv_checkpoints", type=utils.str2intlist, default=[10, 25, 50])
    
    # lambda_ms_ssim
    parser.add_argument("--lambda_ms_ssim", type=float, default=0, help="weight of the MS-SSIM loss")

    
    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
    parser.add_argument("--Shape", type=tuple, default=(3,256,256), help='size of the samples')
    parser.add_argument("--crop_indices", type=int, nargs='+', default=[0,256,0,256])

    ########################## CONTROL of Data to invert ######################
    parser.add_argument("--dates_file", type=str, default = 'Large_lt_test_labels.csv')
    parser.add_argument("--date_start", type=str, default = "2021-07-01")
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
    Means=None
    Maxs=None
    Mins=None
    if params.normalization=="meanmax":
        Means = np.load(f'{params.real_data_dir}{params.mean_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
        Maxs = np.load(f'{params.real_data_dir}{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    elif params.normalization=="minmax":
       Mins = np.load(f'{params.real_data_dir}/stat_files/{params.min_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
       Maxs = np.load(f'{params.real_data_dir}/stat_files/{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    else:
       raise ValueError(f"Unknown normalization: {params.normalization}")

    ################ loading network #################
    if params.encoder_framework_type in ['pSp', "restyle-pSp"]:
        network = pSp(config=params).to(params.device)
    elif params.encoder_framework_type in ['e4e', "restyle-e4e"]:
        network = e4e(config=params).to(params.device)
    elif params.encoder_framework_type == 'inDomain':
        network = inDomain(config=params).to(params.device)
    elif params.encoder_framework_type == 'FeatureStyle':
        network = FeatureStyleModule(config=params).to(params.device)

    ########### write inversion parameters to file ############
    config_file = params.output_dir + "hybrid_inversion_params.yaml"
    print("writing params config file:", config_file)
    try:
        file=open(config_file,"w")
        yaml.dump(params.__dict__,file)
    except Exception as e:
         print("unable to write params config file")
         print(e)

    # print inversion parameters
    print("\nEncoder Inversion parameters:")
    for key, value in params.__dict__.items():
        print(f"{key}: {value}")

    #################### main loop ##################
    for date_ in list_dates:
        print(date_)
        datename = date_.strftime('%Y-%m-%d')
        print("\n===========================")
        for lt in params.leadtimes:
            params.date_index = datename
            params.lt_index = lt
            
            # Check if the files already exists (to qave computation time)
            already_exist = []
            if params.pack_dir != '' :
                if os.path.isfile(params.pack_dir+f'Rsemble_{datename}_{lt}.npy'):
                    print(params.pack_dir+f'Rsemble_{datename}_{lt}.npy' + ' Pack already Exist')
                    already_exist.append(True)
                else :
                    print(params.pack_dir+f'Rsemble_{datename}_{lt}.npy' + ' Pack do not Exist')
                    already_exist.append(False)

            if os.path.isfile(params.output_dir+'w_{}_{}.npy'.format(params.date_index,lt)):
                print(params.output_dir+'w_{}_{}.npy'.format(params.date_index,lt) + ' already Exist')
                already_exist.append(True)
            else :
                print(params.output_dir+'w_{}_{}.npy'.format(params.date_index,lt) + ' do not Exist')
                already_exist.append(False)
            if os.path.isfile(params.output_dir+'invertFsemble_{}_{}.npy'.format(params.date_index,lt)):
                print(params.output_dir+'invertFsemble_{}_{}.npy'.format(params.date_index,lt) +' already Exist')
                already_exist.append(True)
            else :
                print(params.output_dir+'invertFsemble_{}_{}.npy'.format(params.date_index,lt) + ' do not Exist')
                already_exist.append(False)

            if np.all(already_exist) :
                print('The inversion was already done for the date {} with leadtime {}. This sample is skipped.'.format(datename,lt))
            else :
                
                if not params.multi_timestep_mode :
                    print('Launching inversion process for the date {} with leadtime {}.'.format(datename,lt))
                    df0 = df_extract[(df_extract['Date']==date_) & (df_extract['LeadTime']==lt-1)]
                    if len(df0)==0:
                        print("# samples: 0")
                        continue
                    Ens_r = utils.load_batch_from_timestamp(
                        df_extract, 
                        date_, 
                        lt-1, 
                        params.real_data_dir, 
                        Shape=params.Shape, 
                        var_indices=params.var_indices,
                        normalization=params.normalization,
                        Means=Means,
                        Mins=Mins,
                        Maxs=Maxs
                        
                    ) #, crop_indices=params.crop_indices)
                    if params.pack_dir :
                        np.save(params.pack_dir+f'Rsemble_{datename}_{lt}.npy', Ens_r.numpy().astype(np.float32))
                    
                else : 
                    Ens_r = utils.load_batch_sequence_from_date(
                        df_extract,
                        date_,
                        params.real_data_dir,
                        concatenate_variable_and_time=params.stack_sample_along_time_and_variable,
                        dt=params.timestep_period,
                        Shape=params.Shape,
                        var_indices=params.var_indices,
                        normalization=params.normalization,
                        Means=Means,
                        Mins=Mins,
                        Maxs=Maxs
                    )
                    if params.pack_dir :
                        np.save(params.pack_dir+f'Rsemble_sequence_{datename}.npy', Ens_r.numpy().astype(np.float32))

                ################ Forwarding encoder #################
                if params.encoder_framework_type  in ['restyle-pSp', "restyle-e4e"]:
                    init_latent = init_latent_restyle(params=params, network=network, Ens_r=Ens_r)
                elif params.encoder_framework_type  in ['e4e', 'pSp']:
                    init_latent = init_latent_psp_e4e(params=params, network=network, Ens_r=Ens_r)
                elif params.encoder_framework_type == 'inDomain':
                    init_latent = init_latent_inDomain(params=params, network=network, Ens_r=Ens_r)
                elif params.encoder_framework_type == 'FeatureStyle':
                    init_latent = init_latent_featureStyle(params=params, network=network, Ens_r=Ens_r)
                else :
                    raise NotImplementedError

                print(init_latent.shape)
                inv.optimize(
                    Ens_r=Ens_r,
                    g_ema=network.decoder,
                    latent_mean=init_latent,
                    device=params.device,
                    params=params,
                    hybrid=True
                )
                











#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs ensemble forecast inversion using a pre-trained StyleGAN2 model.
The inversion process involves optimizing an initial random latent code so that it best represents a real ensemble forecast input.
This code use the optimization based approach and do not use the encoder based approach.

The code uses command-line arguments for setting directories, inversion parameters, and data control parameters.
The inversion is performed for a specified set of dates and lead times, generating latent code representations for real-ensemble data and saving the results.

Please make sure to configure the directory paths, parameters, and other settings based on your specific environment before running the script.

"""
import torch
import argparse
import os
import numpy as np
from collections import OrderedDict
import yaml
import pandas as pd

print('Importing Generator')
from gan.model.stylegan2 import Generator
print('Importing inversion algo')
import inversion.optimization_based.inversion as inv
print('Importing perturbation utils')
import perturbation.utils as utils


torch.manual_seed(42) #reproducibility of runs

if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################
    # Checkpoint directory - PATH to generator's weight
    parser.add_argument('--ckpt_dir', type = str, 
                        default ='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt')
    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str, 
                        default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, 
                        default ='/project/scratch/p200177/DE_371/victorsanchez/results/member_inversion/test_lpips/inversion_squeeze_tuning/')
    # Pack Directory - PATH where the packed ensembles will be saved
    parser.add_argument("--pack_dir", type=str, 
                        default = '/project/scratch/p200177/DE_371/victorsanchez/results/member_inversion/test_lpips/pack_squeeze_tuning/') # storing "packed" (normalized) real data
    
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

    ############################ INVERSION PARAMETERS #################    

    parser.add_argument("--lr_rampup",type=float,default=0.05,help="duration of the learning rate warmup")
    parser.add_argument("--lr_rampdown",type=float, default=0.25,help="duration of the learning rate decay")
    
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    
    parser.add_argument("--noise_strength", type=float, default=0.005, help="strength of the noise level")
    parser.add_argument("--noise_ramp",type=float,default=0.75,help="duration of the noise level decay")
    
    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
    parser.add_argument("--Shape", type=tuple, default=(3,256,256), help='size of the samples')
    parser.add_argument("--crop_indices", type=int, nargs='+', default=[0,256,0,256])
    
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
    parser.add_argument("--lambda_pixel", type=float, default=10.0, help="weight of the (mae/mse) pixel loss")
    
    # Parameter related to perceptual loss 
    parser.add_argument("--optimize_features_computation", action='store_true', help="Compute the features of original ensemble only once")

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

    parser.add_argument("--invstep", type=int, default=2000, help="optimize iterations")
    parser.add_argument("--inv_checkpoints", type=utils.str2intlist, default=[10,50,100,250,500,1000,1500,2000])
    parser.add_argument("--plot_checkpoint", action='store_true')
    
    # lambda_ms_ssim
    parser.add_argument("--lambda_ms_ssim", type=float, default=0, help="weight of the MS-SSIM loss")

    ########################## CONTROL of Data to invert ######################
    parser.add_argument("--dates_file", type=str, default = 'Large_lt_test_labels.csv')
    parser.add_argument("--date_start", type=str, default = "2021-07-01")
    parser.add_argument("--date_stop", type=str, default = "2021-07-02")
    parser.add_argument("--leadtimes", type=utils.str2intlist, default=[3,6,9,12,15,18,21,24,27,30,33,36,39,42])
    
    parser.add_argument("--seed", type=int, default=42)
    
    params = parser.parse_args()


    # fix some of the inputs
    params.Shape = tuple(params.Shape)
    params.crop_indices = tuple(params.crop_indices)

    # create output and pack directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    if not os.path.exists(params.pack_dir) and params.pack_dir:
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
    #    Means = np.load(f'{params.real_data_dir}stat_files/{params.mean_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    #    Maxs = np.load(f'{params.real_data_dir}/stat_files/{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    elif params.normalization=="minmax":
       Mins = np.load(f'{params.real_data_dir}/stat_files/{params.min_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
       Maxs = np.load(f'{params.real_data_dir}/stat_files/{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    else:
       raise ValueError(f"Unknown normalization: {params.normalization}")

    ################ loading network #################
    if not params.multi_timestep_mode :
        G = Generator(params.Shape[1], 512,n_mlp=8, nb_var=params.Shape[0])
    else :
        G = Generator(params.Shape[1], 512,n_mlp=8, nb_var=params.nb_timesteps)
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

    ########### write inversion parameters to file ############
    config_file = params.output_dir + "inversion_params.yaml"
    print("writing params config file:", config_file)
    try:
        file=open(config_file,"w")
        yaml.dump(params.__dict__,file)
    except Exception as e:
         print("unable to write params config file")
         print(e)

    # print inversion parameters
    print("\nInversion parameters:")
    for key, value in params.__dict__.items():
        print(f"{key}: {value}")

    if params.vgg_loss_after_step >= params.invstep:
        print('The parameters vgg_loss_after_step cannot be superior or equal to the number of optim steps')
        raise ValueError

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
            if os.path.isfile(params.pack_dir+f'Rsemble_{datename}_{lt}.npy'):
                already_exist.append(True)
            else :
                already_exist.append(False)
            for i in params.inv_checkpoints :
                if os.path.isfile(params.output_dir+'w_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i)):
                    already_exist.append(True)
                else :
                    already_exist.append(False)
                if os.path.isfile(params.output_dir+'invertFsemble_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i)):
                    already_exist.append(True)
                else :
                    already_exist.append(False)
                # if os.path.isfile(params.output_dir+'noise_{}_{}_{}.p'.format(params.date_index,params.lt_index,i)):
                #     already_exist.append(True)
                # else :
                #     already_exist.append(False)

            if np.all(already_exist) :
                print('The inversion was already done for the date {} with leadtime {}. This sample is skipped.'.format(datename,lt))
            else :
                print('Launching inversion process for the date {} with leadtime {}.'.format(datename,lt))
                df0 = df_extract[(df_extract['Date']==date_) & (df_extract['LeadTime']==lt-1)]
                if len(df0)==0:
                   print("# samples: 0")
                   continue
                
                if not params.multi_timestep_mode :
                    Ens_r = utils.load_batch_from_timestamp(
                        df_extract, 
                        date_, 
                        lt, 
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

                
                inv.optimize(Ens_r, G, latent_mean, params.device, params)












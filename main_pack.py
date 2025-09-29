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
import utils.utils as utils
from ast import literal_eval as make_tuple
torch.manual_seed(42) #reproducibility of runs

if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()

    ########################### Directories ###########################
    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str,default='')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, default='')
    # Pack Directory - PATH where the packed ensembles will be saved

    parser.add_argument("--pack_dir", type=str, default = '') # storing "packed" (normalized) real data
    parser.add_argument('--ckpt_dir', type = str, default ='')
 
    # Dataset information
    parser.add_argument("--normalization", type=str, default="meanmax", choices=["minmax", "meanmax", ''])
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
    Means=None
    Maxs=None
    Mins=None
    if params.normalization=="meanmax":
        Means = np.load(f'{params.real_data_dir}{params.mean_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
        Maxs = np.load(f'{params.real_data_dir}{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    elif params.normalization=="minmax":
       Mins = np.load(f'{params.real_data_dir}stat_files/{params.min_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
       Maxs = np.load(f'{params.real_data_dir}stat_files/{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
       Means = np.load(f'{params.real_data_dir}stat_files/{params.mean_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    else:
       raise ValueError(f"Unknown normalization: {params.normalization}")

    ########### write inversion parameters to file ############
    config_file = params.output_dir + f"pack_params.yaml"
    print("writing params config file:", config_file)
    try:
        file=open(config_file,"w")
        yaml.dump(params.__dict__,file)
    except Exception as e:
         print("unable to write params config file")
         print(e)


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

            if np.all(already_exist) :
                print('The inversion was already done for the date {} with leadtime {}. This sample is skipped.'.format(datename,lt))
            else :
                
                if not params.multi_timestep_mode :
                    print('Launching inversion process for the date {} with leadtime {}.'.format(datename,lt))
                    df0 = df_extract[(df_extract['Date']==date_) & (df_extract['LeadTime']==lt-1)]
                    if len(df0)==0:
                        print("# samples: 0")
                        continue
                    
                    Ens_r, Ens_r_norm = utils.load_batch_from_timestamp(
                        df_extract, 
                        date_, 
                        lt-1, 
                        params.real_data_dir, 
                        Shape=params.Shape, 
                        var_indices=params.var_indices,
                        normalization=params.normalization,
                        Means=Means,
                        Mins=Mins,
                        Maxs=Maxs,
                        apply_log_transform=True if params.Shape[0]==4 else False
                        
                    ) #, crop_indices=params.crop_indices)                   
                    if params.pack_dir != "" :
                        if params.save_normalized_sample:
                            np.save(params.pack_dir+f'Rsemble_{datename}_{lt}.npy', Ens_r_norm.numpy().astype(np.float32))
                        else :    
                            np.save(params.pack_dir+f'Rsemble_{datename}_{lt}.npy', Ens_r.numpy().astype(np.float32))

import torchvision
import  torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import argparse
import os
import numpy as np
from collections import OrderedDict
import yaml
import pandas as pd

import utils.utils as utils

# TODO : Saliency map with discriminator

if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()
    
    
    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str, 
                        default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, 
                        default ='/project/scratch/p200177/DE_371/victorsanchez/results/feature_importance/')
    parser.add_argument('--inv_dir_encoder', type = str, 
                        default='/project/home/p200177/DE_371/experiments_WP1/encoder_inversion/exp1/inversion/')
    parser.add_argument('--inv_dir_optim', type = str, 
                        default='/project/home/p200177/DE_371/experiments_WP1/inversion_process_analysis/final_inversion_on_test_set/perceptual_exp45/inversion/')
    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
    parser.add_argument("--Shape", type=tuple, default=(3,256,256), help='size of the samples')
    parser.add_argument("--crop_indices", type=int, nargs='+', default=[0,256,0,256])
    
    # Dataset information
    parser.add_argument("--normalization", type=str, default="meanmax", choices=["minmax", "meanmax"])
    parser.add_argument('--max_file', type=str, default='MaxNew_4_var.npy') # use 'MaxNew_4_var.npy' if AROME data # max_rr_log.npy
    parser.add_argument('--mean_file', type=str, default='Mean_4_var.npy') # not used if minmax normalization
    parser.add_argument('--min_file', type=str, default='min_rr_log.npy')  # not used if meanmax normalization
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multi_timestep_mode', action='store_true')
    ########################## CONTROL of Data to invert ######################
    parser.add_argument("--dates_file", type=str, default = 'Large_lt_test_labels.csv')
    parser.add_argument("--date_start", type=str, default = "2021-07-01")
    parser.add_argument("--date_stop", type=str, default = "2021-07-02")
    parser.add_argument("--leadtimes", type=utils.str2intlist, default=[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45]) # ,
    
    parser.add_argument("--seed", type=int, default=42)
    
    params = parser.parse_args()

    # create output and pack directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    # fix some of the inputs
    params.Shape = tuple(params.Shape)
    params.crop_indices = tuple(params.crop_indices)

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

    #load pretrained resnet model
    model = torchvision.models.vgg16(weights=None).to(params.device)
    model.load_state_dict(torch.load('/project/home/p200177/DE_371/resources/network_for_perceptual_loss/vgg16_random.pth'))
    feature_layers = [4,9,16,23,30]
    feature_weighting = [100,100,50,10,10] 
    blocks = []
    blocks.append(model.features[:feature_layers[0]].eval())
    for id in range(len(feature_layers)-1):
        blocks.append(model.features[:feature_layers[id+1]].eval())
    blocks = nn.Sequential(*blocks)

    nb_sample_total = len(list_dates)*len(params.leadtimes)*16
    latent_diff = np.zeros((nb_sample_total, 3, len(blocks), 2))
    #################### main loop ##################
    for id_date, date_ in enumerate(list_dates):
        print(date_)
        datename = date_.strftime('%Y-%m-%d')
        print("\n===========================")
        for id_lt, lt in enumerate(params.leadtimes):
            params.date_index = datename
            params.lt_index = lt
            
            # Check if the files already exists (to qave computation time)
            already_exist = []


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
            
            inv_x_encoder = torch.from_numpy(np.load(params.inv_dir_encoder + f'invertFsemble_{datename}_{lt}_10.npy')).to(params.device)
            inv_x_optim = torch.from_numpy(np.load(params.inv_dir_optim + f'invertFsemble_{datename}_{lt}_1000.npy')).to(params.device)

            x = Ens_r.to(params.device)
            
            for id_member in range(len(x)):
                inv_sample_encoder = inv_x_encoder[id_member].unsqueeze(0)  
                inv_sample_optim = inv_x_optim[id_member].unsqueeze(0) 
                sample = x[id_member].unsqueeze(0)
                for j in range(len(blocks)):
                    #forward pass to calculate predictions
                    for i in range(3):
                        input = sample[:,i,:,:].repeat(1, 3, 1, 1)
                        preds = blocks[j]((input+1)/2)
                        input_inv_encoder = inv_sample_encoder[:,i,:,:].repeat(1, 3, 1, 1)
                        preds_inv_encoder = blocks[j]((input_inv_encoder+1)/2)
                        loss = torch.nn.functional.l1_loss(preds, preds_inv_encoder)
                        latent_diff[id_member+id_date+id_lt,i,j, 0] = loss
                        
                        input_inv_optim = inv_sample_optim[:,i,:,:].repeat(1, 3, 1, 1)
                        preds_inv_optim = blocks[j]((input_inv_optim+1)/2)
                        loss = torch.nn.functional.l1_loss(preds, preds_inv_optim)
                        latent_diff[id_member+id_date+id_lt,i,j, 1] = loss

    latent_diff = np.mean(latent_diff, axis=0)   
    fig, ax = plt.subplots(figsize=(18,10), nrows=3, ncols=1)
    ax[0].plot(range(len(blocks)), latent_diff[0,:], linewidth=10)
    ax[0].set_xticks(range(len(feature_layers)), labels=feature_layers)
    ax[1].plot(range(len(blocks)), latent_diff[1,:], linewidth=10)
    ax[1].set_xticks(range(len(feature_layers)), labels=feature_layers)
    ax[2].plot(range(len(blocks)), latent_diff[2,:], linewidth=10)
    ax[2].set_xticks(range(len(feature_layers)), labels=feature_layers)
    figname = params.output_dir+f'features_importance_avg_over_{nb_sample_total}_random_vgg.png'
    fig.savefig(figname, dpi=100)

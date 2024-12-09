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

import perturbation.utils as utils

# TODO : Saliency map with discriminator

if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()
    
    
    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str, 
                        default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, 
                        default ='/project/scratch/p200177/DE_371/victorsanchez/results/saliency_map/vgg16/')
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
    parser.add_argument("--leadtimes", type=utils.str2intlist, default=[3]) # ,6,9,12,15,18,21,24,27,30,33,36,39,42,45
    
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
    model.load_state_dict(torch.load('/project/home/p200177/DE_371/resources/network_for_perceptual_loss/vgg16_trained.pth'))
    feature_layers = [4,9,16,23,30] 
    blocks = []
    blocks.append(model.features[:feature_layers[0]].eval())
    for id in range(len(feature_layers)-1):
        blocks.append(model.features[:feature_layers[id+1]].eval())
    blocks = nn.Sequential(*blocks)

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

            x = Ens_r[0].to(params.device).unsqueeze(0)
            # for i, block in enumerate(blocks):
            #     x.requires_grad = True
            #     #forward pass to calculate predictions
            #     x = block((x+1)/2)
            #     score, indices = torch.max(x, 1)
            #     
            #     #backward pass to get gradients of score predicted class w.r.t. input image
            #     score.backward()
            #     #get max along channel axis
            #     slc, _ = torch.max(torch.abs(x.grad[0]), dim=0)
            #     #normalize to [0..1]
            #     slc = (slc - slc.min())/(slc.max()-slc.min())

            #     fig, ax = plt.subplots(figsize=(18,10), nrows=2, ncols=3)
            #     ax[0][0].imshow(x[0].detach().numpy(), origin="lower")
            #     ax[0][1].imshow(x[1].detach().numpy(), origin="lower")
            #     ax[0][2].imshow(x[2].detach().numpy(), origin="lower", cmap="coolwarm")
            #     ax[1][0].imshow(slc[0].detach().numpy(), origin="lower", cmap=plt.cm.hot)
            #     ax[1][1].imshow(slc[1].detach().numpy(), origin="lower", cmap=plt.cm.hot)
            #     ax[1][2].imshow(slc[2].detach().numpy(), origin="lower", cmap=plt.cm.hot)
            #     figname = params.output_dir+f'saliency_map_layer_{feature_layers[i]}.png'
            #     fig.savefig(figname, dpi=100)
            x.requires_grad = True
            for j in range(len(blocks)):
                saliencies = []
                #forward pass to calculate predictions
                for i in range(3):
                    input = x[:,i,:,:].repeat(1, 3, 1, 1)
                    preds = blocks[j]((input+1)/2)
                    score, indices = torch.max(preds, 1)

                    #backward pass to get gradients of score predicted class w.r.t. input image
                    score.sum().backward()
                    
                    #get max along channel axis
                    slc, _ = torch.max(torch.abs(x.grad[0]), dim=0)
                    #normalize to [0..1]
                    slc = (slc - slc.min())/(slc.max()-slc.min())
                    saliencies.append(slc)
                img = x.squeeze(0)
                fig, ax = plt.subplots(figsize=(18,10), nrows=2, ncols=3)
                ax[0][0].imshow(img[0].cpu().detach().numpy(), origin="lower")
                ax[0][1].imshow(img[1].cpu().detach().numpy(), origin="lower")
                ax[0][2].imshow(img[2].cpu().detach().numpy(), origin="lower", cmap="coolwarm")
                ax[1][0].imshow(saliencies[0].cpu().detach().numpy(), origin="lower", cmap=plt.cm.hot)
                ax[1][1].imshow(saliencies[1].cpu().detach().numpy(), origin="lower", cmap=plt.cm.hot)
                ax[1][2].imshow(saliencies[2].cpu().detach().numpy(), origin="lower", cmap=plt.cm.hot)
                figname = params.output_dir+f'saliency_map_layer_{feature_layers[j]}_trained_vgg16.png'
                fig.savefig(figname, dpi=100)

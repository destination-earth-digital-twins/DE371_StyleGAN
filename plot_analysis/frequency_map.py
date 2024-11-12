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
import scipy.fftpack as fftpack
import perturbation.utils as utils

# TODO : Saliency map with discriminator

if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()
    
    
    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str, 
                        default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    
    parser.add_argument('--inv_dir', type = str, 
                        default='/project/home/p200177/DE_371/experiments_WP1/encoder_inversion/exp1/inversion/')

    
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, 
                        default ='/project/scratch/p200177/DE_371/victorsanchez/results/frequency_map_test/')
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

    #################### main loop ##################
    for date_ in list_dates:
        print(date_)
        datename = date_.strftime('%Y-%m-%d')
        print("\n===========================")
        for lt in params.leadtimes:
            params.date_index = datename
            params.lt_index = lt


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

            x = Ens_r[0]
            # freq_x = [torch.log10(torch.abs(torch.fft.fft2(x[i], norm='ortho'))).type(torch.float32) for i in range(len(x))]
            
            inv_x = torch.from_numpy(np.load(params.inv_dir + f'invertFsemble_{datename}_{lt}_10.npy'))[0]
            # freq_inv_x = [torch.log10(torch.abs(torch.fft.fft2(inv_x[i], norm='ortho'))).type(torch.float32) for i in range(len(x))]

            fig, ax = plt.subplots(figsize=(18,18), nrows=7, ncols=3)
            fft2_x = [fftpack.fftshift(fftpack.fft2((x[i].numpy())).astype(np.float32)) for i in range(3)]
            fft2_inv_x = [fftpack.fftshift(fftpack.fft2((inv_x[i].numpy())).astype(np.float32)) for i in range(3)]


            freq_x = [(20*np.log10(0.1+fft2_x[i])).astype(np.int8) for i in range(3)]
            freq_inv_x = [(20*np.log10(0.1+fft2_inv_x[i])).astype(np.int8) for i in range(3)]

            
            (_, w, h) = x.shape
            half_w, half_h = int(w/2), int(h/2)

            # # high pass filter
            # n = 5
            # for i in range(3):
            #     freq_x[i][half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 0
            #     freq_inv_x[i][half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 0
            # high_passed_x = np.array([fftpack.ifft2(fftpack.ifftshift(freq_x[i])).real for i in range(3)])

            # # low pass filter
            # n = 120
            # freq_x_low_passed = np.zeros(x.shape)
            # for i in range(3):
            #     freq_x_low_passed[i][half_w-n:half_w+n+1,half_h-n:half_h+n+1] = freq_x[i][half_w-n:half_w+n+1,half_h-n:half_h+n+1]
            #     # freq_inv_x[i][half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 0
      
            diff_of_fft = np.array([fftpack.ifft2(fftpack.ifftshift(fft2_inv_x[i]-fft2_x[i])).real for i in range(3)])  

            # FFT(Diff) or Diff(FFT) ?
            ax[0][0].imshow(x[0].numpy(), origin='lower', clim=(x.min(), x.max()))
            ax[0][1].imshow(x[1].numpy(), origin='lower', clim=(x.min(), x.max()))
            ax[0][2].imshow(x[2].numpy(), origin='lower', clim=(x.min(), x.max()), cmap="coolwarm")
            ax[1][0].imshow(freq_x[0], clim=(freq_x[0].min(), freq_x[0].max()), cmap='Greys')
            ax[1][1].imshow(freq_x[1], clim=(freq_x[1].min(), freq_x[1].max()), cmap='Greys')
            ax[1][2].imshow(freq_x[2], clim=(freq_x[2].min(), freq_x[2].max()), cmap='Greys')
            ax[2][0].imshow(inv_x[0], origin='lower', clim=(x.min(), x.max()))
            ax[2][1].imshow(inv_x[1], origin='lower', clim=(x.min(), x.max()))
            ax[2][2].imshow(inv_x[2], origin='lower', clim=(x.min(), x.max()), cmap="coolwarm")
            ax[3][0].imshow(freq_inv_x[0], clim=(freq_x[0].min(), freq_x[0].max()), cmap='Greys')
            ax[3][1].imshow(freq_inv_x[1], clim=(freq_x[1].min(), freq_x[1].max()), cmap='Greys')
            ax[3][2].imshow(freq_inv_x[2], clim=(freq_x[2].min(), freq_x[2].max()), cmap='Greys')
            ax[4][0].imshow(inv_x[0]-x[0].numpy(), cmap="RdYlGn", origin='lower', clim=(-0.1, 0.1))
            ax[4][1].imshow(inv_x[1]-x[1].numpy(), cmap="RdYlGn", origin='lower', clim=(-0.1, 0.1))
            ax[4][2].imshow(inv_x[2]-x[2].numpy(), cmap="RdYlGn", origin='lower', clim=(-0.1, 0.1))
            ax[5][0].imshow(freq_inv_x[0]-freq_x[0], cmap="PRGn")
            ax[5][1].imshow(freq_inv_x[1]-freq_x[1], cmap="PRGn")
            ax[5][2].imshow(freq_inv_x[2]-freq_x[2], cmap="PRGn")
            ax[6][0].imshow(diff_of_fft[0], cmap="RdYlGn", origin='lower')
            ax[6][1].imshow(diff_of_fft[1], cmap="RdYlGn", origin='lower')
            ax[6][2].imshow(diff_of_fft[2], cmap="RdYlGn", origin='lower')
            figname = params.output_dir+f'frequency_map_layer_{datename}_{lt}.png'
            fig.tight_layout()
            fig.savefig(figname, dpi=100)
            plt.close()

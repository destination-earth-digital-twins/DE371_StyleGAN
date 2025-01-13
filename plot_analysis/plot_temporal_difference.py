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
import matplotlib.pyplot as plt
from datetime import date, timedelta
matplotlib.use('Agg')
import utils.utils as utils

torch.manual_seed(42) #reproducibility of runs

def daterange(start_date: date, end_date: date):
    days = int((end_date - start_date).days)
    for n in range(days):
        yield start_date + timedelta(n)

if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()
    ########################### Directories ###########################

    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str, default='')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, default ='')
    # Pack Directory - PATH where the packed ensembles will be saved
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
    parser.add_argument("--Shape", type=tuple, default=(3,256,256), help='size of the samples')
    parser.add_argument("--crop_indices", type=int, nargs='+', default=[0,256,0,256])

    ########################## CONTROL of Data to invert ######################
    parser.add_argument("--dates_file", type=str, default = 'Large_lt_test_labels.csv')
    parser.add_argument("--date_start", type=str, default = "2021-10-02")
    parser.add_argument("--date_stop", type=str, default = "2021-10-03")
    parser.add_argument("--nb_leadtime", type=int, default=45)
    parser.add_argument("--delta_leadtime", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    
    params = parser.parse_args()


    # fix some of the inputs
    params.Shape = tuple(params.Shape)
    params.crop_indices = tuple(params.crop_indices)

    # create output and pack directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)

    # set the seed for reproducibility of runs
    seed = params.seed
    torch.manual_seed(seed)

    ################## loading dates and file names ##
    df = pd.read_csv(params.real_data_dir + params.dates_file)
    df_date = df.copy()
    df_date['Date'] = pd.to_datetime(df_date['Date'])
    df_extract = df_date[(df_date['Date']>=params.date_start) & (df_date['Date']<=params.date_stop)]

    list_dates = df_extract['Date'].unique()
    

    ########### write parameters to file ############
    config_file = params.output_dir + f"params.yaml"
    print("writing params config file:", config_file)
    try:
        file=open(config_file,"w")
        yaml.dump(params.__dict__,file)
    except Exception as e:
         print("unable to write params config file")
         print(e)

    # print inversion parameters
    print("\n Inversion parameters:")
    for key, value in params.__dict__.items():
        print(f"{key}: {value}")

    #################### main loop ##################
    id = 0
    for id_date, date_ in enumerate(list_dates):
        print(date_)
        datename = date_.strftime('%Y-%m-%d')
        print("\n===========================")
        for lt in range(1,params.nb_leadtime, params.delta_leadtime):
            params.date_index = datename
            params.lt_index = lt
            filename = 'Delta_sample_{}'.format('{num:0{width}}'.format(num=id, width=6))

            
            Ens_r_t, _ = utils.load_batch_from_timestamp(
                df_extract, 
                date_, 
                lt-1, 
                params.real_data_dir, 
                Shape=params.Shape, 
                var_indices=params.var_indices,
                normalization=''
            )

            Ens_r_t_next, _ = utils.load_batch_from_timestamp(
                df_extract, 
                date_, 
                lt-1+params.delta_leadtime, 
                params.real_data_dir, 
                Shape=params.Shape, 
                var_indices=params.var_indices,
                normalization=''
            )

            Delta_Ens = Ens_r_t_next.numpy()-Ens_r_t.numpy()
            Abs_Delta_Ens = np.abs(Delta_Ens)
            Ens_r_t=Ens_r_t.numpy()
            Ens_r_t_next=Ens_r_t_next.numpy()
            for mem_idx in range(1):
                fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(50,50))
                var_names=['u','v','t2m']
                dict_var={'u': 0, 'v': 1, 't2m': 2}
                colormap_var=['viridis','viridis','coolwarm']
                for id, var in enumerate(var_names):
                    var_id = dict_var[var]
                    vmin = np.min([np.min(Ens_r_t[:,var_id])])
                    vmax = np.min([np.max(Ens_r_t[:,var_id])])

                    ax[0][id].set_title(f"field {var} at t",fontsize=70)
                    im = ax[0][id].imshow(Ens_r_t[mem_idx,var_id], clim=(vmin, vmax), origin="lower", cmap=colormap_var[id])
                    cbar = fig.colorbar(im, ax=ax[0][id], shrink=0.5)
                    cbar.ax.tick_params(labelsize=50)

                    ax[1][id].set_title(f"field {var} at t+{params.delta_leadtime}",fontsize=70)
                    im = ax[1][id].imshow(Ens_r_t_next[mem_idx,var_id], clim=(vmin, vmax), origin="lower", cmap=colormap_var[id])
                    cbar = fig.colorbar(im, ax=ax[1][id], shrink=0.5)
                    cbar.ax.tick_params(labelsize=50)

                    # ax[2][id].set_title(f"Delta field {var}",fontsize=70)
                    # im = ax[2][id].imshow(Delta_Ens[mem_idx,var_id], origin="lower", cmap="RdYlGn")
                    # im.set_clim(-10,10)
                    # fig.colorbar(im, ax=ax[2][id], shrink=0.5)  
                    
                    ax[2][id].set_title(f"Abs Delta field {var}",fontsize=70)
                    im = ax[2][id].imshow(Abs_Delta_Ens[mem_idx,var_id], origin="lower", cmap="Greens")
                    im.set_clim(0,10)
                    cbar = fig.colorbar(im, ax=ax[2][id], shrink=0.5)  
                    cbar.ax.tick_params(labelsize=50)

                fig.suptitle(f'Fields at time t={lt}, t+{params.delta_leadtime} and corresponding Delta field - {datename}-{mem_idx}', fontsize=90)  
                fig.tight_layout()
                fig.savefig(params.output_dir+f'Temporal_Difference_{datename}_{mem_idx}_{lt}_Delta_{params.delta_leadtime}.png')
                plt.close()

            
        
    
            









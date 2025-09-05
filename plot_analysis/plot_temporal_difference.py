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
import matplotlib.pyplot as plt
torch.manual_seed(42) #reproducibility of runs

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
    

    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str,default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, default='/project/home/p200177/DE_371/experiments_WP1/GAN_temporal_experiments_AROME/plots/')
    # Pack Directory - PATH where the packed ensembles will be saved 

    # Dataset information
    parser.add_argument("--normalization", type=str, default="minmax")
    parser.add_argument('--max_file', type=str, default='max_rr_log.npy') # use 'MaxNew_4_var.npy' if AROME data # max_rr_log.npy
    parser.add_argument('--mean_file', type=str, default='mean_rr_log.npy') # not used if minmax normalization
    parser.add_argument('--min_file', type=str, default='min_rr_log.npy')  # not used if meanmax normalization
    parser.add_argument('--save_normalized_sample', action='store_true')

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
    parser.add_argument("--Shape", type=make_tuple, default=(3,256,256), help='size of the samples')
    parser.add_argument("--crop_indices", type=int, nargs='+', default=[0,256,0,256])

    ########################## CONTROL of Data to invert ######################
    parser.add_argument("--dates_file", type=str, help='csv file')
    parser.add_argument("--date_start", type=str, default = "2021-07-01")
    parser.add_argument("--date_stop", type=str, default = "2021-07-02")
    
    
    parser.add_argument("--seed", type=int, default=42)
    
    params = parser.parse_args()
    params.leadtimes = list(range(45))

    # fix some of the inputs
    params.Shape = tuple(params.Shape)
    params.crop_indices = tuple(params.crop_indices)

    # create output and pack directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)

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

    ########### write inversion parameters to file ############
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
    for date_ in list_dates:
        print(date_)
        datename = date_.strftime('%Y-%m-%d')
        Ens_r = utils.load_batch_sequence_from_date(
            df_extract,
            date_,
            params.real_data_dir,
            concatenate_variable_and_time=False,
            dt=1,
            Shape=params.Shape,
            var_indices=params.var_indices,
            normalization=params.normalization,
            Means=Means,
            Mins=Mins,
            Maxs=Maxs,
            apply_log_transform=True if params.Shape[0]==4 else False
        )
        var_names=['u','v','t2m']
        dict_var={'u': 0, 'v': 1, 't2m': 2},
        clim_var={'u': [-5,5], 'v': [-5,5], 't2m': [-5,5]}
        colormap_var=['viridis','viridis','coolwarm']
        mem_idx=0
        
        for i in range(len(Ens_r)-1):
            x = Ens_r[i]
            x_next = Ens_r[i+1]
            fig, ax = plt.subplots(figsize=(15,5*len(var_names)), nrows=3, ncols=len(var_names))
            figtitle=f'Temporal difference from {datename}_{params.leadtimes[i]}'
            for id, var in enumerate(var_names):

                var_id = id
                ax[0][id].set_title(f"{var}(t)")
                im = ax[0][id].imshow(x[mem_idx,var_id], origin="lower", cmap=colormap_var[id])
                fig.colorbar(im, ax=ax[0][id], shrink=0.5)

                
                im = ax[1][id].imshow(x_next[mem_idx,var_id], origin="lower",  cmap=colormap_var[id])
                ax[1][id].set_title(f"{var}(t+1h)")
                fig.colorbar(im, ax=ax[1][id], shrink=0.5)

                
                diff = x_next[mem_idx,var_id] - x[mem_idx,var_id]
                im = ax[2][id].imshow(diff, origin="lower", cmap="Greens")
                im.set_clim(-0.5,0.5)
                ax[2][id].set_title(f"{var}(t+1h)-{var}(t)")
                fig.colorbar(im, ax=ax[2][id], shrink=0.5)

            fig.suptitle(figtitle)
            fig.tight_layout()
            fig.savefig(params.output_dir+f'Temporal_diff_{datename}_{params.leadtimes[i]}.png', dpi=100)
            plt.close()
            











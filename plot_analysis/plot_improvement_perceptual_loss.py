# Function to plot the perturbated samples from the non temporal gan

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import torch 
import argparse
import utils.utils as utils
import os 
import pandas as pd
import scipy

if __name__=="__main__" :
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################
    
    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str,  default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    parser.add_argument('--pack_dir', type = str,  default='/project/home/p200177/DE_371/experiments_WP1/inversion_process_analysis/final_inversion_on_test_set/pack_meanmax/')
    parser.add_argument('--inv_dir_mse', type = str,  default='/project/home/p200177/DE_371/experiments_WP1/inversion_process_analysis/final_inversion_on_test_set/perceptual_mse_exp39/inversion/')
    parser.add_argument('--inv_dir_perceptual', type = str,  default='/project/home/p200177/DE_371/experiments_WP1/inversion_process_analysis/final_inversion_on_test_set/perceptual_exp45/inversion/')

    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, default ='/project/home/p200177/DE_371/experiments_WP1/inversion_process_analysis/final_inversion_on_test_set/visual_comparison/')
    parser.add_argument("--Shape", type=tuple, default=(3,256,256), help='size of the samples')
    # Dataset information
    parser.add_argument("--normalization", type=str, default="meanmax", choices=["minmax", "meanmax"])
    parser.add_argument('--max_file', type=str, default='MaxNew_4_var.npy') # use 'MaxNew_4_var.npy' if AROME data # max_rr_log.npy
    parser.add_argument('--mean_file', type=str, default='Mean_4_var.npy') # not used if minmax normalization
    parser.add_argument('--min_file', type=str, default='min_rr_log.npy')  # not used if meanmax normalization
    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
    parser.add_argument("--var_names", type=utils.str2intlist, default=['u','v','t2m'])
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--sample_from_imported_perturbation', action='store_true')
    ############################ SEQUENCE PARAMETERS #################    
    parser.add_argument('--multi_timestep_mode', action='store_true')
    parser.add_argument('--nb_timesteps', type=int, default=14)
    parser.add_argument('--timestep_period', type=int, default=3)
    parser.add_argument('--stack_sample_along_time_and_variable', action='store_true')

    ############################ INVERSION PARAMETERS ################
    parser.add_argument("--invstep", type=int, default=1000, help="optimize iterations")

    ########################## CONTROL of Data to invert ######################
    parser.add_argument("--dates_file", type=str, default = 'Large_lt_test_labels.csv')
    parser.add_argument("--date_start", type=str, default = "2020-10-01")
    parser.add_argument("--date_stop", type=str, default = "2022-10-02")
    parser.add_argument("--leadtimes", type=utils.str2intlist, default=[3,6,9,12,15,18,21,24,27,30,33,36,39,42])
    
    parser.add_argument("--seed", type=int, default=42)
    
    params = parser.parse_args()

    # fix some of the inputs
    params.Shape = tuple(params.Shape)
    
    output_dir = params.output_dir
    # create output and pack directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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


    Arome_sample = None
    Mse_inv_sample = None
    Perceptual_inv_sample = None
    for id_date, date_ in enumerate(list_dates):
        date_=str(date_)[:10]
        for lt in params.leadtimes :
            try :
                path_to_sample = params.pack_dir + f"Rsemble_{date_}_{lt}.npy"    
                Arome_sample = np.load(path_to_sample)
            except :
                print(params.pack_dir + f"File Rsemble_{date_}_{lt}.npy Not found")
            Arome_sample = utils.rescale(Arome_sample, Means, Maxs, 1/0.95)

            try :
                path_to_sample = params.inv_dir_mse + f"invertFsemble_{date_}_{lt}_2000.npy"    
                Mse_inv_sample = np.load(path_to_sample)
            except :
                print(params.inv_dir_mse + f"invertFsemble_{date_}_{lt}_2000.npy Not found")
            Mse_inv_sample = utils.rescale(Mse_inv_sample, Means, Maxs, 1/0.95)

            try :
                path_to_sample = params.inv_dir_perceptual + f"invertFsemble_{date_}_{lt}_1000.npy"    
                Perceptual_inv_sample = np.load(path_to_sample)
            except :
                print(params.inv_dir_perceptual + f"invertFsemble_{date_}_{lt}_1000.npy Not found")
            Perceptual_inv_sample = utils.rescale(Perceptual_inv_sample, Means, Maxs, 1/0.95)

            fig = plt.figure(figsize=(15,15))
            mem_idx = 0
            # U
            vmin = np.min([np.min(Arome_sample[mem_idx,0])])
            vmax = np.min([np.max(Arome_sample[mem_idx,0])])

            ax = fig.add_subplot(331)
            ax.set_title("u real")
            im = ax.imshow(Arome_sample[mem_idx,0], clim=(vmin, vmax), origin="lower")
            fig.colorbar(im, shrink=0.5)

            ax = fig.add_subplot(334)
            ax.set_title("u inv (MSE)")
            im = ax.imshow(Mse_inv_sample[mem_idx,0], clim=(vmin, vmax), origin="lower")
            fig.colorbar(im, shrink=0.5)

            ax = fig.add_subplot(337)
            ax.set_title("u inv (Perceptual)")
            im = ax.imshow(Perceptual_inv_sample[mem_idx,0], clim=(vmin, vmax), origin="lower")
            fig.colorbar(im, shrink=0.5)

            # V
            vmin = np.min([np.min(Arome_sample[mem_idx,1])])
            vmax = np.min([np.max(Arome_sample[mem_idx,1])])
            
            ax = fig.add_subplot(332)
            ax.set_title("v real")
            im = ax.imshow(Arome_sample[mem_idx,1], clim=(vmin, vmax), origin="lower")
            fig.colorbar(im, shrink=0.5)

            ax = fig.add_subplot(335)
            ax.set_title("v inv (MSE)")
            im = ax.imshow(Mse_inv_sample[mem_idx,1], clim=(vmin, vmax), origin="lower")
            fig.colorbar(im, shrink=0.5)

            ax = fig.add_subplot(338)
            ax.set_title("v inv (Perceptual)")
            im = ax.imshow(Perceptual_inv_sample[mem_idx,1], clim=(vmin, vmax), origin="lower")
            fig.colorbar(im, shrink=0.5)

            # t2m
            vmin = np.min([np.min(Arome_sample[mem_idx,2])])
            vmax = np.min([np.max(Arome_sample[mem_idx,2])])

            ax = fig.add_subplot(333)
            ax.set_title("t2m real")  
            im = ax.imshow(Arome_sample[mem_idx,2], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
            fig.colorbar(im, shrink=0.5)

            ax = fig.add_subplot(336)
            ax.set_title("t2m inv (MSE)")
            im = ax.imshow(Mse_inv_sample[mem_idx,2], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
            fig.colorbar(im, shrink=0.5)

            ax = fig.add_subplot(339)
            ax.set_title("t2m inv (Perceptual)")
            im = ax.imshow(Perceptual_inv_sample[mem_idx,2], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
            fig.colorbar(im, shrink=0.5)

            figtitle = ''
            fig_name = output_dir + f'Comparison_MSE_Perceptual_{date_}_{lt}.png'
            fig.suptitle(figtitle)
            fig.tight_layout()
            fig.savefig(fig_name, dpi=100)

            plt.close()
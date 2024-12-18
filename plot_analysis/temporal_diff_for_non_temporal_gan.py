# Function to plot the perturbated samples from the non temporal gan

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import torch 
import argparse
import utils.utils as utils
import os 
import pandas as pd

if __name__=="__main__" :
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################
    
    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str,  default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, default ='/project/scratch/p200177/DE_371/victorsanchez/results/temporal_analysis/spatial_temporal_difference/')
    parser.add_argument('--gen_sample_dir',type = str, default ="/project/scratch/p200177/DE_371/victorsanchez/results/perturbation/coherence_temporelle_gan_classique/stochastic_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_False_-1_16_/samples/")
    parser.add_argument("--Shape", type=tuple, default=(3,256,256), help='size of the samples')
    # Dataset information
    parser.add_argument("--normalization", type=str, default="meanmax", choices=["minmax", "meanmax"])
    parser.add_argument('--max_file', type=str, default='MaxNew_4_var.npy') # use 'MaxNew_4_var.npy' if AROME data # max_rr_log.npy
    parser.add_argument('--mean_file', type=str, default='Mean_4_var.npy') # not used if minmax normalization
    parser.add_argument('--min_file', type=str, default='min_rr_log.npy')  # not used if meanmax normalization
    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
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
    parser.add_argument("--date_start", type=str, default = "2021-10-01")
    parser.add_argument("--date_stop", type=str, default = "2021-10-03")
    parser.add_argument("--leadtimes", type=utils.str2intlist, default=[3,6,9,12,15,18,21,24,27,30,33,36,39,42])
    
    parser.add_argument("--seed", type=int, default=42)
    
    params = parser.parse_args()

    # fix some of the inputs
    params.Shape = tuple(params.Shape)
    
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
    if params.normalization=="meanmax":
        Means = np.load(f'{params.real_data_dir}{params.mean_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
        Maxs = np.load(f'{params.real_data_dir}{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    elif params.normalization=="minmax":
        Mins = np.load(f'{params.real_data_dir}/stat_files/{params.min_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
        Maxs = np.load(f'{params.real_data_dir}/stat_files/{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    else:
        raise ValueError(f"Unknown normalization: {params.normalization}")

    display_AROME = True
    if not params.sample_from_imported_perturbation:
        label_perturbation = '_generated_pert'
    else :
        label_perturbation = '_imported_pert'
    #################### main loop ##################
    for date_ in list_dates:
        # Importing True samples
        if display_AROME:
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
            Ens_r = utils.rescale(Ens_r.numpy(), Means, Maxs, 1/0.95)
        # Importing Generated Samples
        Ens_gen = []
        date_=str(date_)[:10]
        for lt in params.leadtimes :
            try :
                path_to_sample = params.gen_sample_dir + f"genFsemble_{date_}_{lt}_{params.invstep}_16{label_perturbation}.npy"    
                Ens_gen.append(np.load(path_to_sample))
            except :
                print(f"File 'genFsemble_{date_}_{lt}_{params.invstep}_16{label_perturbation}.npy' Not Found")
        Ens_gen = np.array(Ens_gen)
        Ens_gen = utils.rescale(Ens_gen, Means, Maxs, 1/0.95)
        
        
        Nb_member=16
        groud_length = 4
        for member_id in range(16):
            fig0, ax0 = plt.subplots(nrows=groud_length, ncols=13, figsize=(200,50))
            fig1, ax1 = plt.subplots(nrows=groud_length, ncols=13, figsize=(200,50))
            fig2, ax2 = plt.subplots(nrows=groud_length, ncols=13, figsize=(200,50))
            for t in trange(params.nb_timesteps-1):
                ims = []
                for ax_id, ax in enumerate([ax0, ax1, ax2]):  
                    Arome_member = Ens_r[member_id][t+1] - Ens_r[member_id][t]
                    im=ax[0][t].imshow(Arome_member[ax_id], origin="lower", cmap="RdYlGn", vmin=Arome_member[ax_id].min(), vmax=Arome_member[ax_id].max())
                    ims.append(im)
                    ax[0][t].set_ylabel(f'Arome M{member_id}-{t}', fontsize=45)
                    ax[0][t].set_xticks([])
                    ax[0][t].set_yticks([])

                    for i in range(1,groud_length):
                        generated_member_id = i*16+member_id
                        Generated_member = Ens_gen[t+1][generated_member_id] - Ens_gen[t][generated_member_id]
                        ax[i][t].imshow(Generated_member[ax_id], origin="lower", cmap="RdYlGn", vmin=Arome_member[ax_id].min(), vmax=Arome_member[ax_id].max())
                        ax[i][t].set_ylabel(f'Gen M{generated_member_id}-t+{t}', fontsize=45)
                        ax[i][t].set_xticks([])
                        ax[i][t].set_yticks([])
                
                if t==0:

                    fig0.suptitle(f"Temporal Difference u(t+1)-u(t) for {date_}", fontsize=100)
                    fig1.suptitle(f"Temporal Difference v(t+1)-v(t) for {date_}", fontsize=100)
                    fig2.suptitle(f"Temporal Difference t2m(t+1)-t2m(t) for {date_}", fontsize=100)
                

                    for fig,im in zip([fig0,fig1,fig2],ims):
                        fig.subplots_adjust(bottom=0.05,top=0.9, left=0.05, right=0.9)
                        cbax=fig.add_axes([0.92,0.05,0.02,0.85])
                        cb=fig.colorbar(im, cax=cbax)
                        cb.ax.tick_params(labelsize=80) 
                        # fig.tight_layout()

            if display_AROME:
                fig0.savefig(params.output_dir+f'Temporal_Difference_u_{date_}_{member_id}.png')
                fig1.savefig(params.output_dir+f'Temporal_Difference_v_{date_}_{member_id}.png')
                fig2.savefig(params.output_dir+f'Temporal_Difference_t2m_{date_}_{member_id}.png')
            

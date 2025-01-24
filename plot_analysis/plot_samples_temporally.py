# Function to plot the perturbated samples from the non temporal gan

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import torch 
import argparse
import utils.utils as utils
import os 
import pandas as pd
from math import ceil

if __name__=="__main__" :
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################
    
    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str,  default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, default ='/project/home/p200177/DE_371/experiments_WP1/temporal_diff_samples/fixed_perturbation/')
    parser.add_argument('--gen_sample_dir',type = str, default ="/project/home/p200177/DE_371/experiments_WP1/temporal_diff_samples/fixed_perturbation/stochastic_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_False_-1_16_/samples/")
    parser.add_argument("--Shape", type=tuple, default=(3,256,256), help='size of the samples')
    # Dataset information
    parser.add_argument("--normalization", type=str, default="", choices=["minmax", "meanmax", ""])
    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
    parser.add_argument('--device', type=str, default='cuda')

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
    parser.add_argument("--date_stop", type=str, default = "2021-10-02")
    parser.add_argument("--leadtimes", type=utils.str2intlist, default=[3,6,9,12,15,18,21,24,27,30,33,36,39,42])
    
    parser.add_argument("--seed", type=int, default=42)
    
    params = parser.parse_args()

    # fix some of the inputs
    params.Shape = tuple(params.Shape)
    
    params.output_dir+='plot_samples/'
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

    display_temporal_difference = True
    #################### main loop ##################
    for date_ in list_dates:
        # Importing True samples
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
        ).numpy()
        # Importing Generated Samples
        Ens_gen = []
        date_=str(date_)[:10]
        for lt in params.leadtimes :
            try :
                path_to_sample = params.gen_sample_dir + f"genFsemble_{date_}_{lt}_{params.invstep}_16.npy"   
                Ens_gen.append(np.load(path_to_sample))
            except :
                print(f"File 'genFsemble_{date_}_{lt}_{params.invstep}_16.npy' Not Found")
        Ens_gen = np.array(Ens_gen)

        Nb_cond_member = int(ceil(Ens_gen.shape[1] / Ens_r.shape[1]))
        print(f'There are {Nb_cond_member} Child member per Parent member')

        nb_generated_member_to_plot = 3
        for member_id in range(5):
            fig0, ax0 = plt.subplots(nrows=nb_generated_member_to_plot+1, ncols=14, figsize=(200,50))
            fig1, ax1 = plt.subplots(nrows=nb_generated_member_to_plot+1, ncols=14, figsize=(200,50))
            fig2, ax2 = plt.subplots(nrows=nb_generated_member_to_plot+1, ncols=14, figsize=(200,50))
            if display_temporal_difference:
                fig0dt, ax0dt = plt.subplots(nrows=nb_generated_member_to_plot+1, ncols=13, figsize=(200,50))
                fig1dt, ax1dt = plt.subplots(nrows=nb_generated_member_to_plot+1, ncols=13, figsize=(200,50))
                fig2dt, ax2dt = plt.subplots(nrows=nb_generated_member_to_plot+1, ncols=13, figsize=(200,50))

            for t in trange(0, params.nb_timesteps):
                Arome_member = Ens_r[member_id][t+1]
                vmin = [Arome_member[0].min(), Arome_member[1].min(), Arome_member[2].min()]
                vmax = [Arome_member[0].max(), Arome_member[1].max(), Arome_member[2].max()]
                
                im0=ax0[0][t].imshow(Arome_member[0], origin="lower", cmap="viridis", vmin=vmin[0], vmax=vmax[0])
                ax0[0][t].set_ylabel(f'PM:{member_id}-t+{1+t*3}', fontsize=45)
                ax0[0][t].set_xticks([])
                ax0[0][t].set_yticks([])

                im1=ax1[0][t].imshow(Arome_member[1], origin="lower", cmap="viridis", vmin=vmin[1], vmax=vmax[1])
                ax1[0][t].set_ylabel(f'PM:{member_id}-t+{1+t*3}', fontsize=45)
                ax1[0][t].set_xticks([])
                ax1[0][t].set_yticks([])

                im2=ax2[0][t].imshow(Arome_member[2], origin="lower", cmap="coolwarm", vmin=vmin[2], vmax=vmax[2])
                ax2[0][t].set_ylabel(f'PM:{member_id}-t+{1+t*3}', fontsize=45)
                ax2[0][t].set_xticks([])
                ax2[0][t].set_yticks([])
                if display_temporal_difference and t < params.nb_timesteps-1:
                    Arome_member_dt = Ens_r[member_id][t+2] - Ens_r[member_id][t+1]
                    vmin_dt = [Arome_member_dt[0].min(), Arome_member_dt[1].min(), Arome_member_dt[2].min()]
                    vmax_dt = [Arome_member_dt[0].max(), Arome_member_dt[1].max(), Arome_member_dt[2].max()]

                    im0dt=ax0dt[0][t].imshow(Arome_member_dt[0], origin="lower", cmap="RdYlGn", vmin=vmin_dt[0], vmax=vmax_dt[0])
                    ax0dt[0][t].set_ylabel(f'PM:{member_id}-t+{1+t*3}', fontsize=45)
                    ax0dt[0][t].set_xticks([])
                    ax0dt[0][t].set_yticks([])

                    im1dt=ax1dt[0][t].imshow(Arome_member_dt[1], origin="lower", cmap="RdYlGn", vmin=vmin_dt[1], vmax=vmax_dt[1])
                    ax1dt[0][t].set_ylabel(f'PM:{member_id}-t+{1+t*3}', fontsize=45)
                    ax1dt[0][t].set_xticks([])
                    ax1dt[0][t].set_yticks([])

                    im2dt=ax2dt[0][t].imshow(Arome_member_dt[2], origin="lower", cmap="RdYlGn", vmin=vmin_dt[2], vmax=vmax_dt[2])
                    ax2dt[0][t].set_ylabel(f'PM:{member_id}-t+{1+t*3}', fontsize=45)
                    ax2dt[0][t].set_xticks([])
                    ax2dt[0][t].set_yticks([])

                for offset_id in range(0, nb_generated_member_to_plot):
                    Generated_member = Ens_gen[t][offset_id+Nb_cond_member*member_id]

                    im0=ax0[1+offset_id][t].imshow(Generated_member[0], origin="lower", cmap="viridis", vmin=vmin[1], vmax=vmax[1])
                    ax0[1+offset_id][t].set_ylabel(f'CM:{offset_id+Nb_cond_member*member_id}-t+{1+t*3}', fontsize=45)
                    ax0[1+offset_id][t].set_xticks([])
                    ax0[1+offset_id][t].set_yticks([])

                    im1=ax1[1+offset_id][t].imshow(Generated_member[1], origin="lower", cmap="viridis", vmin=vmin[1], vmax=vmax[1])
                    ax1[1+offset_id][t].set_ylabel(f'CM:{offset_id+Nb_cond_member*member_id}-t+{1+t*3}', fontsize=45)
                    ax1[1+offset_id][t].set_xticks([])
                    ax1[1+offset_id][t].set_yticks([])

                    im2=ax2[1+offset_id][t].imshow(Generated_member[2], origin="lower", cmap="coolwarm", vmin=vmin[2], vmax=vmax[2])
                    ax2[1+offset_id][t].set_ylabel(f'CM:{offset_id+Nb_cond_member*member_id}-t+{1+t*3}', fontsize=45)
                    ax2[1+offset_id][t].set_xticks([])
                    ax2[1+offset_id][t].set_yticks([])
                
                    if display_temporal_difference and t < params.nb_timesteps-1:
                        Generated_member_dt = Ens_gen[t+1][offset_id*member_id+member_id] - Ens_gen[t][offset_id*member_id+member_id]
                        
                        im0dt=ax0dt[1+offset_id][t].imshow(Generated_member_dt[0], origin="lower", cmap="RdYlGn", vmin=vmin_dt[0], vmax=vmax_dt[0])
                        ax0dt[1+offset_id][t].set_ylabel(f'CM:{offset_id+Nb_cond_member*member_id}-t+{1+t*3}', fontsize=45)
                        ax0dt[1+offset_id][t].set_xticks([])
                        ax0dt[1+offset_id][t].set_yticks([])

                        im1dt=ax1dt[1+offset_id][t].imshow(Generated_member_dt[1], origin="lower", cmap="RdYlGn", vmin=vmin_dt[1], vmax=vmax_dt[1])
                        ax1dt[1+offset_id][t].set_ylabel(f'CM:{offset_id+Nb_cond_member*member_id}-t+{1+t*3}', fontsize=45)
                        ax1dt[1+offset_id][t].set_xticks([])
                        ax1dt[1+offset_id][t].set_yticks([])

                        im2dt=ax2dt[1+offset_id][t].imshow(Generated_member_dt[2], origin="lower", cmap="RdYlGn", vmin=vmin_dt[2], vmax=vmax_dt[2])
                        ax2dt[1+offset_id][t].set_ylabel(f'CM:{offset_id+Nb_cond_member*member_id}-t+{1+t*3}', fontsize=45)
                        ax2dt[1+offset_id][t].set_xticks([])
                        ax2dt[1+offset_id][t].set_yticks([])
                        
                
                if t==0 :
                    fig0.suptitle(f"Sequence of u for {date_}", fontsize=200)
                    fig1.suptitle(f"Sequence of v for {date_}", fontsize=200)
                    fig2.suptitle(f"Sequence of t2m for {date_}", fontsize=200)
                    if display_temporal_difference:
                        fig0dt.suptitle(f"Temporal Difference of u for {date_}", fontsize=200)
                        fig1dt.suptitle(f"Temporal Difference of v for {date_}", fontsize=200)
                        fig2dt.suptitle(f"Temporal Difference of t2m for {date_}", fontsize=200)
                   
                    figlist = ([fig0,fig1,fig2],[im0,im1,im2])
                    if display_temporal_difference:
                        figlist = ([fig0,fig1,fig2, fig0dt,fig1dt,fig2dt],[im0,im1,im2, im0dt,im1dt,im2dt])
                
                    for fig,im in zip(figlist[0], figlist[1]):
                        fig.subplots_adjust(bottom=0.05,top=0.9, left=0.05, right=0.9)
                        cbax=fig.add_axes([0.92,0.05,0.02,0.85])
                        cb=fig.colorbar(im, cax=cbax)
                        cb.ax.tick_params(labelsize=80) 
                        # fig.tight_layout()

            fig0.savefig(params.output_dir+f'sequence_u_{date_}_member_{member_id}.png')
            fig1.savefig(params.output_dir+f'sequence_v_{date_}_member_{member_id}.png')
            fig2.savefig(params.output_dir+f'sequence_t2m_{date_}_member_{member_id}.png')
            if display_temporal_difference:
                fig0dt.savefig(params.output_dir+f'temporal_difference_u_{date_}_member_{member_id}.png')
                fig1dt.savefig(params.output_dir+f'temporal_difference_v_{date_}_member_{member_id}.png')
                fig2dt.savefig(params.output_dir+f'temporal_difference_t2m_{date_}_member_{member_id}.png')
        


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

    display_AROME = True
    display_temporal_difference = True
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
        
        
        Nb_member=16
        groud_length = 4
        member_id_list = list(range(0,Nb_member+1,groud_length))
        for group_id in range(len(member_id_list)-2):
            start_member_id = member_id_list[group_id]
            end_member_id = member_id_list[group_id+1]
            print(start_member_id, end_member_id)
            if display_AROME:
                fig0, ax0 = plt.subplots(nrows=groud_length, ncols=14, figsize=(150,50))
                fig1, ax1 = plt.subplots(nrows=groud_length, ncols=14, figsize=(150,50))
                fig2, ax2 = plt.subplots(nrows=groud_length, ncols=14, figsize=(150,50))
                if display_temporal_difference:
                    fig0dt, ax0dt = plt.subplots(nrows=groud_length, ncols=13, figsize=(150,50))
                    fig1dt, ax1dt = plt.subplots(nrows=groud_length, ncols=13, figsize=(150,50))
                    fig2dt, ax2dt = plt.subplots(nrows=groud_length, ncols=13, figsize=(150,50))
            fig0gen, ax0gen = plt.subplots(nrows=groud_length, ncols=14, figsize=(150,50))
            fig1gen, ax1gen = plt.subplots(nrows=groud_length, ncols=14, figsize=(150,50))
            fig2gen, ax2gen = plt.subplots(nrows=groud_length, ncols=14, figsize=(150,50))
            if display_temporal_difference:
                fig0gendt, ax0gendt = plt.subplots(nrows=groud_length, ncols=13, figsize=(150,50))
                fig1gendt, ax1gendt = plt.subplots(nrows=groud_length, ncols=13, figsize=(150,50))
                fig2gendt, ax2gendt = plt.subplots(nrows=groud_length, ncols=13, figsize=(150,50))

            offset=1
            for t in trange(params.nb_timesteps):
                for member_id in range(start_member_id, end_member_id):
                    if display_AROME:
                        Arome_member = Ens_r[offset*member_id+member_id][t]
                        
                        im0=ax0[member_id-start_member_id][t].imshow(Arome_member[0], origin="lower", cmap="viridis", vmin=Arome_member[0].min(), vmax=Arome_member[0].max())
                        ax0[member_id-start_member_id][t].set_ylabel(f'M{offset*(start_member_id+member_id)+(start_member_id+member_id+1)}-t+{t*3}', fontsize=45)
                        ax0[member_id-start_member_id][t].set_xticks([])
                        ax0[member_id-start_member_id][t].set_yticks([])

                        im1=ax1[member_id-start_member_id][t].imshow(Arome_member[1], origin="lower", cmap="viridis", vmin=Arome_member[1].min(), vmax=Arome_member[1].max())
                        ax1[member_id-start_member_id][t].set_ylabel(f'M{offset*(start_member_id+member_id)+(start_member_id+member_id+1)}-t+{t*3}', fontsize=45)
                        ax1[member_id-start_member_id][t].set_xticks([])
                        ax1[member_id-start_member_id][t].set_yticks([])

                        im2=ax2[member_id-start_member_id][t].imshow(Arome_member[2], origin="lower", cmap="coolwarm", vmin=Arome_member[2].min(), vmax=Arome_member[2].max())
                        ax2[member_id-start_member_id][t].set_ylabel(f'M{offset*(start_member_id+member_id)+(start_member_id+member_id+1)}-t+{t*3}', fontsize=45)
                        ax2[member_id-start_member_id][t].set_xticks([])
                        ax2[member_id-start_member_id][t].set_yticks([])
                        if display_temporal_difference and t < params.nb_timesteps-1:
                            Arome_member_dt = Ens_r[offset*member_id+member_id][t+1] - Ens_r[offset*member_id+member_id][t]
                            
                            im0dt=ax0dt[member_id-start_member_id][t].imshow(Arome_member_dt[0], origin="lower", cmap="RdYlGn", vmin=Arome_member_dt[0].min(), vmax=Arome_member_dt[0].max())
                            ax0dt[member_id-start_member_id][t].set_ylabel(f'M{offset*(start_member_id+member_id)+(start_member_id+member_id+1)}-t+{t*3}', fontsize=45)
                            ax0dt[member_id-start_member_id][t].set_xticks([])
                            ax0dt[member_id-start_member_id][t].set_yticks([])

                            im1dt=ax1dt[member_id-start_member_id][t].imshow(Arome_member_dt[1], origin="lower", cmap="RdYlGn", vmin=Arome_member_dt[1].min(), vmax=Arome_member_dt[1].max())
                            ax1dt[member_id-start_member_id][t].set_ylabel(f'M{offset*(start_member_id+member_id)+(start_member_id+member_id+1)}-t+{t*3}', fontsize=45)
                            ax1dt[member_id-start_member_id][t].set_xticks([])
                            ax1dt[member_id-start_member_id][t].set_yticks([])

                            im2dt=ax2dt[member_id-start_member_id][t].imshow(Arome_member_dt[2], origin="lower", cmap="RdYlGn", vmin=Arome_member_dt[2].min(), vmax=Arome_member_dt[2].max())
                            ax2dt[member_id-start_member_id][t].set_ylabel(f'M{offset*(start_member_id+member_id)+(start_member_id+member_id+1)}-t+{t*3}', fontsize=45)
                            ax2dt[member_id-start_member_id][t].set_xticks([])
                            ax2dt[member_id-start_member_id][t].set_yticks([])
                    # Generated_member = Ens_gen[t][(offset*member_id+member_id)*7] # We multiply by 7 to keep child members from the same father member
                    Generated_member = Ens_gen[t][offset*member_id+member_id]
                    
                    im0gen=ax0gen[member_id-start_member_id][t].imshow(Generated_member[0], origin="lower", cmap="viridis", vmin=Generated_member[0].min(), vmax=Generated_member[0].max())
                    ax0gen[member_id-start_member_id][t].set_ylabel(f'GEN M{offset*(start_member_id+member_id)+(start_member_id+member_id+1)}-t+{t*3}', fontsize=45)
                    ax0gen[member_id-start_member_id][t].set_xticks([])
                    ax0gen[member_id-start_member_id][t].set_yticks([])

                    im1gen=ax1gen[member_id-start_member_id][t].imshow(Generated_member[1], origin="lower", cmap="viridis", vmin=Generated_member[1].min(), vmax=Generated_member[1].max())
                    ax1gen[member_id-start_member_id][t].set_ylabel(f'GEN M{offset*(start_member_id+member_id)+(start_member_id+member_id+1)}-t+{t*3}', fontsize=45)
                    ax1gen[member_id-start_member_id][t].set_xticks([])
                    ax1gen[member_id-start_member_id][t].set_yticks([])

                    im2gen=ax2gen[member_id-start_member_id][t].imshow(Generated_member[2], origin="lower", cmap="coolwarm", vmin=Generated_member[2].min(), vmax=Generated_member[2].max())
                    ax2gen[member_id-start_member_id][t].set_ylabel(f'GEN M{offset*(start_member_id+member_id)+(start_member_id+member_id+1)}-t+{t*3}', fontsize=45)
                    ax2gen[member_id-start_member_id][t].set_xticks([])
                    ax2gen[member_id-start_member_id][t].set_yticks([])
                    if display_temporal_difference and t < params.nb_timesteps-1:
                        Generated_member_dt = Ens_gen[t+1][offset*member_id+member_id] - Ens_gen[t][offset*member_id+member_id]

                        im0gendt=ax0gendt[member_id-start_member_id][t].imshow(Generated_member_dt[0], origin="lower", cmap="RdYlGn", vmin=Generated_member_dt[0].min(), vmax=Generated_member_dt[0].max())
                        ax0gendt[member_id-start_member_id][t].set_ylabel(f'GEN M{offset*(start_member_id+member_id)+(start_member_id+member_id+1)}-t+{t*3}', fontsize=45)
                        ax0gendt[member_id-start_member_id][t].set_xticks([])
                        ax0gendt[member_id-start_member_id][t].set_yticks([])

                        im1gendt=ax1gendt[member_id-start_member_id][t].imshow(Generated_member_dt[1], origin="lower", cmap="RdYlGn", vmin=Generated_member_dt[1].min(), vmax=Generated_member_dt[1].max())
                        ax1gendt[member_id-start_member_id][t].set_ylabel(f'GEN M{offset*(start_member_id+member_id)+(start_member_id+member_id+1)}-t+{t*3}', fontsize=45)
                        ax1gendt[member_id-start_member_id][t].set_xticks([])
                        ax1gendt[member_id-start_member_id][t].set_yticks([])

                        im2gendt=ax2gendt[member_id-start_member_id][t].imshow(Generated_member_dt[2], origin="lower", cmap="RdYlGn", vmin=Generated_member_dt[2].min(), vmax=Generated_member_dt[2].max())
                        ax2gendt[member_id-start_member_id][t].set_ylabel(f'GEN M{offset*(start_member_id+member_id)+(start_member_id+member_id+1)}-t+{t*3}', fontsize=45)
                        ax2gendt[member_id-start_member_id][t].set_xticks([])
                        ax2gendt[member_id-start_member_id][t].set_yticks([])
                
                if t==0 and offset ==0:
                    if display_AROME:
                        fig0.suptitle(f"AROME Sequence of u for {date_}", fontsize=100)
                        fig1.suptitle(f"AROME Sequence of v for {date_}", fontsize=100)
                        fig2.suptitle(f"AROME Sequence of t2m for {date_}", fontsize=100)
                        if display_temporal_difference:
                            fig0dt.suptitle(f"AROME Temporal Difference of u for {date_}", fontsize=100)
                            fig1dt.suptitle(f"AROME Temporal Difference of v for {date_}", fontsize=100)
                            fig2dt.suptitle(f"AROME Temporal Difference of t2m for {date_}", fontsize=100)
                    fig0gen.suptitle(f"Generated Samples of u for {date_}", fontsize=100)
                    fig1gen.suptitle(f"Generated Samples of v for {date_}", fontsize=100)
                    fig2gen.suptitle(f"Generated Samples of t2m for {date_}", fontsize=100)
                    if display_temporal_difference:
                        fig0gendt.suptitle(f"Temporal Difference of Generated Samples of u for {date_}", fontsize=100)
                        fig1gendt.suptitle(f"Temporal Difference of Generated Samples of v for {date_}", fontsize=100)
                        fig2gendt.suptitle(f"Temporal Difference of Generated Samples of t2m for {date_}", fontsize=100)

                    figlist = ([fig0,fig1,fig2],[im0,im1,im2])
                    if display_AROME:
                        figlist = ([fig0,fig1,fig2, fig0gen, fig1gen, fig2gen],[im0,im1,im2, im0gen, im1gen, im2gen])
                        if display_temporal_difference:
                            figlist = ([fig0,fig1,fig2, fig0gen, fig1gen, fig2gen, fig0dt,fig1dt,fig2dt, fig0gendt, fig1gendt, fig2gendt],[im0,im1,im2, im0gen, im1gen, im2gen, im0dt,im1dt,im2dt, im0gendt, im1gendt, im2gendt])
                    if display_temporal_difference:
                        figlist = ([fig0gen, fig1gen, fig2gen, fig0gendt, fig1gendt, fig2gendt],[im0gen, im1gen, im2gen, im0gendt, im1gendt, im2gendt])

                    for fig,im in zip(figlist[0], figlist[1]):
                        fig.subplots_adjust(bottom=0.05,top=0.9, left=0.05, right=0.9)
                        cbax=fig.add_axes([0.92,0.05,0.02,0.85])
                        cb=fig.colorbar(im, cax=cbax)
                        cb.ax.tick_params(labelsize=80) 
                        # fig.tight_layout()

            if display_AROME:
                fig0.savefig(params.output_dir+f'AROME_sequence_u_{date_}_{Nb_member}_{start_member_id}_{end_member_id}_{offset}.png')
                fig1.savefig(params.output_dir+f'AROME_sequence_v_{date_}_{Nb_member}_{start_member_id}_{end_member_id}_{offset}.png')
                fig2.savefig(params.output_dir+f'AROME_sequence_t2m_{date_}_{start_member_id}_{end_member_id}_{offset}.png')
                if display_temporal_difference:
                    fig0dt.savefig(params.output_dir+f'AROME_temporal_difference_u_{date_}_{Nb_member}_{start_member_id}_{end_member_id}_{offset}.png')
                    fig1dt.savefig(params.output_dir+f'AROME_temporal_difference_v_{date_}_{Nb_member}_{start_member_id}_{end_member_id}_{offset}.png')
                    fig2dt.savefig(params.output_dir+f'AROME_temporal_difference_t2m_{date_}_{start_member_id}_{end_member_id}_{offset}.png')
            fig0gen.savefig(params.output_dir+f'gen_samples_u_{date_}_{Nb_member}_{start_member_id}_{end_member_id}_{offset}.png')
            fig1gen.savefig(params.output_dir+f'gen_samples_v_{date_}_{Nb_member}_{start_member_id}_{end_member_id}_{offset}.png')
            fig2gen.savefig(params.output_dir+f'gen_samples_t2m_{date_}_{Nb_member}_{start_member_id}_{end_member_id}_{offset}.png')
            if display_temporal_difference:
                fig0gendt.savefig(params.output_dir+f'gen_samples_temporal_difference_u_{date_}_{Nb_member}_{start_member_id}_{end_member_id}_{offset}.png')
                fig1gendt.savefig(params.output_dir+f'gen_samples_temporal_difference_v_{date_}_{Nb_member}_{start_member_id}_{end_member_id}_{offset}.png')
                fig2gendt.savefig(params.output_dir+f'gen_samples_temporal_difference_t2m_{date_}_{Nb_member}_{start_member_id}_{end_member_id}_{offset}.png')
                # plt.close()


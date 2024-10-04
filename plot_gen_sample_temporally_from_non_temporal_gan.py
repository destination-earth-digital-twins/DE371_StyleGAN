# Function to plot the perturbated samples from the non temporal gan

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import torch 
import argparse
import perturbation.utils as utils
import os 
import pandas as pd

if __name__=="__main__" :
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################
    
    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str,  default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, default ='/project/scratch/p200177/DE_371/victorsanchez/results/perturbation/coherence_temporelle_gan_classique/plots/')
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
    parser.add_argument("--invstep", type=int, default=2000, help="optimize iterations")

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

    display_AROME = False
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
                Ens_gen.append(np.load(path_to_sample)[list(range(2,112,7))])
            except :
                print(f"File 'genFsemble_{date_}_{lt}_{params.invstep}_16{label_perturbation}.npy' Not Found")
        Ens_gen = np.array(Ens_gen)
        Ens_gen = utils.rescale(Ens_gen, Means, Maxs, 1/0.95)
        
        Nb_member=16
        if display_AROME:
            fig0, ax0 = plt.subplots(nrows=Nb_member, ncols=14, figsize=(200,200))
            fig1, ax1 = plt.subplots(nrows=Nb_member, ncols=14, figsize=(200,200))
            fig2, ax2 = plt.subplots(nrows=Nb_member, ncols=14, figsize=(200,200))
        fig0gen, ax0gen = plt.subplots(nrows=Nb_member, ncols=14, figsize=(200,200))
        fig1gen, ax1gen = plt.subplots(nrows=Nb_member, ncols=14, figsize=(200,200))
        fig2gen, ax2gen = plt.subplots(nrows=Nb_member, ncols=14, figsize=(200,200))

        for t in trange(params.nb_timesteps):
            for member_id in range(Nb_member):
                if display_AROME:
                    Arome_member = Ens_r[member_id][t]
                    

                    im0=ax0[member_id][t].imshow(Arome_member[0], origin="lower", cmap="viridis", vmin=Arome_member[0].min(), vmax=Arome_member[0].max())
                    ax0[member_id][t].set_ylabel(f'M{member_id+1}-t+{t}', fontsize=45)
                    ax0[member_id][t].set_xticks([])
                    ax0[member_id][t].set_yticks([])

                    im1=ax1[member_id][t].imshow(Arome_member[1], origin="lower", cmap="viridis", vmin=Arome_member[1].min(), vmax=Arome_member[1].max())
                    ax1[member_id][t].set_ylabel(f'M{member_id+1}-t+{t}', fontsize=45)
                    ax1[member_id][t].set_xticks([])
                    ax1[member_id][t].set_yticks([])

                    im2=ax2[member_id][t].imshow(Arome_member[2], origin="lower", cmap="coolwarm", vmin=Arome_member[2].min(), vmax=Arome_member[2].max())
                    ax2[member_id][t].set_ylabel(f'M{member_id+1}-t+{t}', fontsize=45)
                    ax2[member_id][t].set_xticks([])
                    ax2[member_id][t].set_yticks([])

                Generated_member = Ens_gen[t][member_id]
                im0gen=ax0gen[member_id][t].imshow(Generated_member[0], origin="lower", cmap="viridis", vmin=Generated_member[0].min(), vmax=Generated_member[0].max())
                ax0gen[member_id][t].set_ylabel(f'GEN M{member_id+1}-t+{t}', fontsize=45)
                ax0gen[member_id][t].set_xticks([])
                ax0gen[member_id][t].set_yticks([])

                im1gen=ax1gen[member_id][t].imshow(Generated_member[1], origin="lower", cmap="viridis", vmin=Generated_member[1].min(), vmax=Generated_member[1].max())
                ax1gen[member_id][t].set_ylabel(f'GEN M{member_id+1}-t+{t}', fontsize=45)
                ax1gen[member_id][t].set_xticks([])
                ax1gen[member_id][t].set_yticks([])

                im2gen=ax2gen[member_id][t].imshow(Generated_member[2], origin="lower", cmap="coolwarm", vmin=Generated_member[2].min(), vmax=Generated_member[2].max())
                ax2gen[member_id][t].set_ylabel(f'GEN M{member_id+1}-t+{t}', fontsize=45)
                ax2gen[member_id][t].set_xticks([])
                ax2gen[member_id][t].set_yticks([])
            
            
            if t==0 :
                if display_AROME:
                    fig0.suptitle(f"AROME Sequence of u for {date_}", fontsize=100)
                    fig1.suptitle(f"AROME Sequence of v for {date_}", fontsize=100)
                    fig2.suptitle(f"AROME Sequence of t2m for {date_}", fontsize=100)
                fig0gen.suptitle(f"Generated Samples of u for {date_}", fontsize=100)
                fig1gen.suptitle(f"Generated Samples of v for {date_}", fontsize=100)
                fig2gen.suptitle(f"Generated Samples of t2m for {date_}", fontsize=100)
                if display_AROME:
                    for fig,im in zip([fig0,fig1,fig2, fig0gen, fig1gen, fig2gen],[im0,im1,im2, im0gen, im1gen, im2gen]):
                        fig.subplots_adjust(bottom=0.05,top=0.9, left=0.05, right=0.9)
                        cbax=fig.add_axes([0.92,0.05,0.02,0.85])
                        cb=fig.colorbar(im, cax=cbax)
                        cb.ax.tick_params(labelsize=80) 
                        # fig.tight_layout()
                else :
                    for fig,im in zip([fig0gen, fig1gen, fig2gen],[im0gen, im1gen, im2gen]):
                        fig.subplots_adjust(bottom=0.05,top=0.9, left=0.05, right=0.9)
                        cbax=fig.add_axes([0.92,0.05,0.02,0.85])
                        cb=fig.colorbar(im, cax=cbax)
                        cb.ax.tick_params(labelsize=80) 
                        # fig.tight_layout()
        if display_AROME:
            fig0.savefig(params.output_dir+f'AROME_sequence_u_{date_}.png')
            fig1.savefig(params.output_dir+f'AROME_sequence_v_{date_}.png')
            fig2.savefig(params.output_dir+f'AROME_sequence_t2m_{date_}.png')
        fig0gen.savefig(params.output_dir+f'gen_samples_u_{date_}{label_perturbation}.png')
        fig1gen.savefig(params.output_dir+f'gen_samples_v_{date_}{label_perturbation}.png')
        fig2gen.savefig(params.output_dir+f'gen_samples_t2m_{date_}{label_perturbation}.png')
        
        plt.close()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:21:37 2023

@author: brochetc

Main pod sampling script

"""

import torch
import argparse
import os
import numpy as np
import pickle
import pandas as pd
from datetime import date, timedelta, datetime
from time import perf_counter
from collections import OrderedDict
from gan.model.stylegan2 import Generator

import perturbation.utils as utils
import perturbation.smpca as smpca
from shutil import copyfile
from inversion.plotter import online_pert_plot


def str2list(li):
    if type(li)==list:
        li2 = li
        return li2
    
    elif type(li)==str:
        li2=li[1:-1].split(',')
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))
        

def compute_generate_save(G, params, metrics_list, Means, Maxs):

    N_samples = params.N_samples
    if params.verbose:
        print(datename, lt)
        print(params.date_index, params.lt_index)
    Ens_r = torch.tensor(np.load(params.pack_dir+f'Rsemble_{datename}_{lt}.npy'), dtype = torch.float32)
    w_ens = torch.tensor(np.load(params.data_dir + f'w_{params.date_index}_{params.lt_index}_{params.inv_step}.npy').astype(np.float32))
    inv_ens=np.load(params.data_dir + f'invertFsemble_{params.date_index}_{params.lt_index}_{params.inv_step}.npy').astype(np.float32)
    # subsampling if N_conditioners is lower than initial ensemble size
    if (params.N_conditioners<w_ens.shape[0]):
        cond_indices = np.random.choice(range(w_ens.shape[0]), params.N_conditioners)
        w_ens = w_ens[cond_indices]
        Ens_r = Ens_r[cond_indices]
    print('############### Perturbating ###############')
    print('loading generation hyperparams')
    Whitening = torch.load(params.eigendir + 'Whitening.pt') if params.sample_rule=='stochastic' else None
    Coloring = torch.load(params.eigendir + 'Coloring.pt') if params.sample_rule=='stochastic' else None
    w0 = torch.load(params.eigendir + 'latent_mean.pt') if params.sample_rule=='stochastic' else None
    scale = torch.tensor(np.load(os.path.join(params.scale_dir,"ema_scale.npy")).astype(np.float32)[params.scale_interp_step], device=params.device)
    interp = torch.tensor(np.load(os.path.join(params.scale_dir,"ema_interp.npy")).astype(np.float32)[params.scale_interp_step], device=params.device)

    gen, w_new = smpca.sm_pca(
        Ens_w=w_ens, 
        G=G, 
        N_samples=N_samples, 
        sm_ind=params.style_indices,
        device=params.device, 
        sample_rule=params.sample_rule, 
        random_unbias=False,
        scale=scale,
        interp=interp,
        verbose=params.verbose,
        Whitening=Whitening,
        Coloring=Coloring,
        w0=w0
    )

    if params.verbose:
        print(gen.mean(axis=(0,-2,-1)))
        print(Ens_r.mean(axis=(0,-2,-1)))
    np.save(params.output_dir + f'/samples/genFsemble_{params.date_index}_{params.lt_index}_{params.inv_step}_{params.N_conditioners}.npy', gen)

    online_pert_plot(
        packsample=Ens_r.numpy(), 
        invsample=inv_ens, 
        pert_sample=gen,
        crop=[0,-1,0,-1],
        mem_idx=0 if params.N_conditioners>1 else cond_indices, 
        figtitle=f"Generated samples for {params.date_index}_{params.lt_index}_{params.inv_step}_{params.N_conditioners}", 
        figname=params.output_dir + f"/samples/genFsemble_{params.date_index}_{params.lt_index}_{params.inv_step}_{params.N_conditioners}.png"
    )

    if params.runtime_metrics:
        gen0 = utils.rescale(gen, Means, Maxs, 1/0.95)
        Ens_r = utils.rescale(Ens_r.detach().cpu().numpy(), Means, Maxs, 1/0.95)
        dic = {'Mean' : {'real':Ens_r.mean(axis=(0,-2,-1)), 'fake':gen0.mean(axis=(0,-2,-1))},
             'Std' : {'real': np.sqrt(Ens_r.var(axis=0).mean(axis=(-2,-1))), 'fake': np.sqrt(gen0.var(axis=0).mean(axis=(-2,-1)))},
             'Max' : {'real':Ens_r.max(axis=(0,-2,-1)), 'fake':gen0.max(axis=(0,-2,-1))},
             'Align' : 1.0 - (gen0.std(axis=0) * Ens_r.std(axis=0)).sum(axis=(-2,-1)) / (np.sqrt((Ens_r.std(axis=0) ** 2).sum(axis=(-2,-1))) *  np.sqrt((gen0.std(axis=0) **2).sum(axis=(-2,-1)))),
        }
        if params.verbose: print(dic)
        return dic
    return {}
     
if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################
    # Checkpoint directory - PATH to generator's weight
    parser.add_argument('--ckpt_dir', type = str, 
                        default ='/home/users/u101833/project/results/models/trained_generator/000024.pt')
    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str, 
                        default='/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    # Data Directory - PATH to samples from inversion process                    
    parser.add_argument('--data_dir', type=str, 
                        default='/home/users/u101833/project/results/inversion/victorsanchez/Ens_Perceptual_Random_VGG_Loss/Inversion_Perceptual_Random_VGG_Loss/') #'/scratch/mrmn/brochetc/GAN_2D/Exp_StyleGAN_final/Inversion_Val/')
    # Pack Directory - PATH where the packed ensembles will be saved
    parser.add_argument("--pack_dir", type=str, 
                        default = '/home/users/u101833/project/results/inversion/victorsanchez/Ens_Perceptual_Random_VGG_Loss/Pack_Perceptual_Random_VGG_Loss/') # storing "packed" (normalized) real data
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, 
                        default ='/home/users/u101833/project/results/victorsanchez/perturbation/')
    parser.add_argument('--path_root_readme',type = str, 
                        default ='/project/scratch/p200177/DE_371/victorsanchez/other/ReadMe_0.txt')

    # Generator network information
    parser.add_argument('--add_name',type = str, default='')
    parser.add_argument('--eigendir',type = str, default ='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/eigenvalues_gan_training/')
    
    
    # Dataset information
    parser.add_argument("--normalization", type=str, default="meanmax", choices=["minmax", "meanmax"])
    parser.add_argument('--max_file', type=str, default='MaxNew_4_var.npy') # use 'MaxNew_4_var.npy' if AROME data # max_rr_log.npy
    parser.add_argument('--mean_file', type=str, default='Mean_4_var.npy') # not used if minmax normalization
    parser.add_argument('--min_file', type=str, default='min_rr_log.npy')  # not used if meanmax normalization

    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
    parser.add_argument("--Shape", type=tuple, default=(3,256,256), help='size of the samples')
    parser.add_argument("--N_samples", type=int, default=10, help='number of new samples') 
    parser.add_argument("--N_conditioners",type=int, default=16, help="number of 'seed' samples used for conditioning")
    parser.add_argument("--inv_step", type=int, default=2000, help='step of inversion to load w')
    
    
    ######################## PERTURBATION PARAMETERS #######################
    parser.add_argument('--sample_rule', type=str, default='stochastic', 
                        choices = ['stochastic', 'extrapolation'])

    parser.add_argument('--style_indices', type = str2list, default='[1,1,1,1,1,1,1,1,1,1,0,0,0,0]')
    parser.add_argument('--unbias', action="store_true")

    parser.add_argument('--scale_dir', type=str, default="/project/home/p200177/DE_371/datasets/dataset_Meteo_France/scale_dir_gan_training/")
    parser.add_argument('--scale_interp_step',type=int, default=-1)
    

    ########################## CONTROL of Data to perturb ######################
    parser.add_argument("--dates_file", type=str, default = 'Large_lt_test_labels.csv')
    parser.add_argument("--date_start", type=str, default = "2021-07-01")
    parser.add_argument("--date_stop", type=str, default = "2021-07-02")
    parser.add_argument("--leadtimes", type=utils.str2intlist, default= [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45])

    ###########################################################################
    parser.add_argument("--runtime_metrics", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument('--device', type=str, default='cuda') # if torch.cuda.is_available() else 'cpu')

    params = parser.parse_args()
    params.output_dir = params.output_dir + f"{params.sample_rule}_{params.style_indices}_{params.unbias}_{params.scale_interp_step}_{params.N_conditioners}_{params.add_name}/" 

    # create output directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)

    ################## selecting dates
    print('reading dates')
    df = pd.read_csv(params.real_data_dir + params.dates_file)
    df_date = df.copy()
    df_date['Date'] = pd.to_datetime(df_date['Date'])
    df_extract = df_date[(df_date['Date']>=params.date_start) & (df_date['Date']<=params.date_stop)]
    liste_dates = df_extract['Date'].unique()
    
    ################## carrying scaling info to pass it whenever needed
    Means = np.load(f'{params.real_data_dir}{params.mean_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    Maxs = np.load(f'{params.real_data_dir}{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    scale = (1/0.95)
    
    ############################################################
    if not os.path.exists(params.output_dir + 'samples/'):
        os.mkdir(params.output_dir + 'samples/')
    if not os.path.exists(params.output_dir + 'log/'):
        os.mkdir(params.output_dir + 'log/')
    source_readme = params.path_root_readme
    target_readme = params.output_dir + 'ReadMe_0.txt'
    copyfile(source_readme, target_readme)
    
    
    ################ loading network #################

    
    print('loading G')
    G = Generator(params.Shape[1], 512,n_mlp=8,nb_var=params.Shape[0])
    print('JE SUIS PARAMS',params.Shape[1],params.Shape[0],params.ckpt_dir )
    ckpt = torch.load(params.ckpt_dir, map_location='cpu')['g_ema']
    if 'module' in list(ckpt.items())[0][0]: #juggling with Pytorch versioning and different module packaging
        ckpt_adapt = OrderedDict()
        for k in ckpt.keys():
            k0 = k[7:]
            ckpt_adapt[k0] = ckpt[k]
        G.load_state_dict(ckpt_adapt)
    else:
        G.load_state_dict(ckpt)
    G.eval()
    G = G.to(params.device)
    print('JE SUIS GENERATOR',G)
    #############################  Main loop ###############################
    
    metrics_list = ['variance', 'std_diff']#, 'mean_bias']
    metrics = {}
    for date_ in liste_dates:
        datename = date_.strftime('%Y-%m-%d')
        for lt in params.leadtimes:
            params.date_index = datename
            params.lt_index = lt
            
            already_exist = False
            if os.path.isfile(params.output_dir + f'/samples/genFsemble_{params.date_index}_{params.lt_index}_{params.inv_step}_{params.N_conditioners}.npy'):
                already_exist=True
            if already_exist :
                print('The perturbation was already done for the date {} with leadtime {}. This sample is skipped.'.format(datename,lt))
            else :
                print('Launching perturbation process for the date {} with leadtime {}.'.format(datename,lt))    
                try:
                    print('generating')
                    metrics[(datename,lt)] = compute_generate_save(G, params, metrics_list, Means, Maxs)
                except FileNotFoundError as e:
                    print(f"File not found {e}")
                    pass
            
    path = params.output_dir + 'log/'
    pickle.dump(metrics,open(path+'metrics.p','wb'))

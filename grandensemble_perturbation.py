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

print('importing network')
from gan.model.stylegan2 import Generator
import metrics4arome as metrics

import perturbation.utils as utils
import perturbation.smpca as smpca
from shutil import copyfile

device = 'cuda:0'

def str2list(li):
    if type(li)==list:
        li2 = li
        return li2
    
    elif type(li)==str:
        li2=li[1:-1].split(',')
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))
        

def compute_generate_save(G, params, metrics_list, Means, Maxs, scale):

    N_samples = params.N_samples

    print('simple load')
    Ens_r = utils.collate_R_ensemble(params.real_data_dir,params.mb,params.lt_index, params.var_indices)
    print('r (rescaled) loaded')
    print(params.mb, params.lt_index)
    
    w_ens = torch.tensor(utils.collate_w_ensemble(params.data_dir, params.mb, params.lt_index, params.var_indices), dtype=torch.float32)
    print('w loaded')
    print('############### Perturbating ###############')
    
    Whitening = torch.load(params.eigendir + 'Whitening.pt') if params.sample_rule=='stochastic' else None
    Coloring = torch.load(params.eigendir + 'Coloring.pt') if params.sample_rule=='stochastic' else None
    w0 = torch.load(params.eigendir + 'latent_mean.pt') if params.sample_rule=='stochastic' else None
    scale = torch.tensor(np.load(os.path.join(params.scale_dir,"ema_scale.npy")).astype(np.float32)[params.scale_interp_step], device=device)
    interp = torch.tensor(np.load(os.path.join(params.scale_dir,"ema_interp.npy")).astype(np.float32)[params.scale_interp_step], device=device)

    gen, w_new = smpca.sm_pca(w_ens, G, 
                         N_samples, 
                         params.style_indices, params.device, params.sample_rule,
                         scale=scale,interp=interp, verbose=True,
                         Whitening=Whitening,Coloring=Coloring,w0=w0)
    
    gen0 = utils.rescale(gen, Means, Maxs, 1/0.95)
    Ens_r = utils.rescale(Ens_r, Means, Maxs, 1/0.95)

    if params.runtime_metrics:
        print('############### Evaluating metrics ###############')
        
        dic_metrics = {}
        
        for m in metrics_list :
            print(m)
            metr = getattr(metrics,m)
            if m in metrics.standalone_metrics :
                dic_metrics[m] = metr(gen0)
            elif m in metrics.distance_metrics :
                dic_metrics[m] = metr(Ens_r, gen0)
            else:
                raise ValueError('Metric unknown')

            pickle.dump(dic_metrics,open(params.output_dir + f'/log/metrics_{params.draw_index}_{params.lt_index}_{params.inv_step}.p', 'wb'))

    np.save(params.output_dir + f'/samples/genFsemble_{params.draw_index}_{params.lt_index}_{params.inv_step}.npy', gen0)

    print(gen.shape)

    
if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################

    parser.add_argument('--ckpt_dir', type = str, 
                        default ='/scratch/work/brochetc/Exp_StyleGAN/Set_1/stylegan2_stylegan_dom_256_lat-dim_512_bs_4_0.002_0.002_ch-mul_2_vars_u_v_t2m_noise_True/Instance_14/models/000024.pt')
    parser.add_argument('--real_data_dir', type = str, 
                        default ='/scratch/work/brochetc/Exp_StyleGAN/Pack_GE/')
    parser.add_argument('--data_dir', type=str, default='/scratch/work/brochetc/Exp_StyleGAN/Inversion_GE/')
    parser.add_argument('--output_dir',type = str, 
                        default ='/scratch/work/brochetc/Exp_StyleGAN/Perturbation_GE/')
    parser.add_argument('--eigendir',type = str, 
                        default ='/scratch/work/brochetc/Exp_StyleGAN/Eigenvalues/')
    parser.add_argument("--pack_dir", type=str, default = '') # storing "packed" (normalized) real data
    
    parser.add_argument('--mean_file', type=str, default='Mean_4_var.npy')
    parser.add_argument('--max_file', type=str, default='MaxNew_4_var.npy')

    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
    parser.add_argument("--Shape", type=tuple, default=(3,256,256), help='size of the samples')
    parser.add_argument("--N_samples", type=int, default=120, help='number of new samples')
    parser.add_argument("--N_draws", type=int, default=300, help='number of conditioning members draws')    
    parser.add_argument("--inv_step", type=int, default=1000, help='step of inversion to load w')
    
    parser.add_argument("--device", type=str, default='cuda:0')

    
    ######################## PERTURBATION PARAMETERS #######################
    
    parser.add_argument('--sample_rule', type=str, default='stochastic', 
                        choices = ['stochastic', 'extrapolation'])
    parser.add_argument('--style_indices', type = str2list, default='[0,0,0,0,0,0,0,0,0,0,0,0,0,0]')
    parser.add_argument('--scale_dir', type=str, default="./")
    parser.add_argument('--scale_interp_step',type=int, default=-1)

    parser.add_argument('--unbias',action='store_true')
    ########################## CONTROL of Data to perturb ######################

    parser.add_argument("--leadtimes", type=utils.str2intlist, default=[3,6,12,18,24,30,36,42])

    ###########################################################################
    parser.add_argument("--runtime_metrics", action="store_true")

    ###########################################################################
    
    params = parser.parse_args()
    root_dir = params.output_dir 
    params.output_dir = params.output_dir + f'{params.sample_rule}_{params.style_indices}_{params.unbias}/' 
    
    N_samples = params.N_samples

    N_draws = params.N_draws
    
    ################## carrying scaling info to pass it whenever needed
    
    Means = np.load(f'{params.real_data_dir}{params.mean_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)

    Maxs = np.load(f'{params.real_data_dir}{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    
    scale = (1/0.95)
    
    ############################################################
    
    if not os.path.exists(params.output_dir) :
        os.mkdir(params.output_dir)
        os.mkdir(params.output_dir + '/samples/')
        os.mkdir(params.output_dir + '/log/')
        source_readme = root_dir + 'ReadMe_0.txt'
        target_readme = params.output_dir + 'ReadMe_0.txt'
        copyfile(source_readme, target_readme)
    
    
    ################ loading network #################

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    G = Generator(params.Shape[1], 512,n_mlp=8,nb_var=params.Shape[0])


    #print('###########################################"##################################################################################################################')
    ckpt = torch.load(params.ckpt_dir, map_location='cpu')['g_ema']

    if 'module' in list(ckpt.items())[0][0]: #juglling with Pytorch versioning and different module packaging
        ckpt_adapt = OrderedDict()
        for k in ckpt.keys():
            k0 = k[7:]
            ckpt_adapt[k0] = ckpt[k]
        G.load_state_dict(ckpt_adapt)


    else:
        G.load_state_dict(ckpt)
    G.eval()

    
    G = G.to(device)


    #############################  Main loop ###############################
    
    metrics_list = ['quantiles', 'variance', 'std_diff', 'mean_bias']
    
    for draw_idx in range(N_draws) : # we make N_draws choices of random 16 members,
        print(f"Drawing {draw_idx}th")
        mb = utils.initsmall().astype(np.uint32)  # select the associated w, and generate new samples from these (with offset for python)

        params.mb = mb
        for lt in params.leadtimes:
            np.save(params.output_dir + f'mb_{draw_idx}_{lt}_1000.npy', np.array(mb))
            print(mb,lt)

            params.lt_index = lt
            params.draw_index = draw_idx
        
            compute_generate_save(G, params, metrics_list, Means, Maxs, scale)
        print(f"Ending {draw_idx}th")


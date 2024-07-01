#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:49:10 2023

@author: brochetc
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from argparse import Namespace

from torch.nn import functional as F
from models.stylegan2.model import Generator
from models.encoders import restyle_psp_encoders
from GradientMagnitude import gradient_magnitude, style_mixing_ensemble, style_mixing_threshold


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

Path = '/scratch/mrmn/brochetc/GAN_2D/style_mixing_expe/real_samples/'
Path_samples = '/scratch/mrmn/brochetc/GAN_2D/style_mixing_expe/Samples/'
Path_out = '/scratch/mrmn/brochetc/GAN_2D/style_mixing_expe/ensemble_comparison/'

Path_ckpt_stylegan = '/scratch/mrmn/brochetc/GAN_2D/style_mixing_expe/'
Path_ckpt_encoder = '/scratch/mrmn/brochetc/GAN_2D/style_mixing_expe/'
df = pd.read_csv(Path + 'IS_method_labels.csv') 

n_styles = 12

def load_and_initialize_stylegan(ckpt) : 
    
    model = Generator(128,512,8, channel_multiplier = 2)
    
    state = torch.load(Path_ckpt_stylegan + ckpt)
    
    model.load_state_dict(state['g_ema'])
    
    return model

def load_and_initialize_encoder(ckpt, opts):
    
    encoder = restyle_psp_encoders.ResNetBackboneEncoder(n_styles, opts)
    
    
    state = torch.load(Path_ckpt_encoder + ckpt)['state_dict']
    
    #print(state.keys())
    
    mapped_encoder_ckpt = dict(state)
       
    for p in state.keys():
        if 'encoder' in p :
            v = state[p]
            mapped_encoder_ckpt.pop(p)
            mapped_encoder_ckpt[p.replace('encoder.', '')] = v
        elif 'decoder' in p :
            mapped_encoder_ckpt.pop(p)
        
    encoder.load_state_dict(mapped_encoder_ckpt)
    
    return encoder




def load_ensemble():
    
    df_M = df.query("Date == '2021-01-21T21:00:00Z'")[df["LeadTime"] == 3]
    #Ens_list = df_M["Name"].tolist()
    Ens_list = list(range(16))
    Len_Ens_list = len(Ens_list)
    
    Ens_member_index = np.zeros(Len_Ens_list, dtype = int)
    
    #for i in range(Len_Ens_list):
    #    Ens_member_index[i] = int(Ens_list[i].replace('_sample', ''))
        
    ##############################################################################
    
    
    #####################LOADING REAL/PROJ ENSEMBLE and W ###############################
    
    Ensemble_real = np.zeros((Len_Ens_list, 3, 128, 128))
    
    for i in range(Len_Ens_list) :
        
        Ensemble_real[i] = np.load(Path + '_sample'+ str(Ens_list[i]) + '.npy').astype(np.float32)[1:4,78:206,55:183]
        
    return torch.tensor(Ensemble_real)

def optimisation_loss():
    return 0

if __name__=="__main__" :    
    
    opts = Namespace(**{'input_nc' : 6})
    
    g_ema = load_and_initialize_stylegan('285000.pt').to(device)
    
    encoder = load_and_initialize_encoder('best_model.pt', opts).to(device)
    
    avg_sample = torch.tensor(np.load(Path_ckpt_encoder + 'avg_sample.pth.npy').astype(np.float32)).float()
    
    Ensemble_real = load_ensemble().float()
    
    X_input = torch.cat([Ensemble_real, avg_sample.repeat(Ensemble_real.shape[0], 1,1,1)], dim = 1).to(device)
    
    with torch.no_grad():
        W_ens = encoder(X_input)
    
    print(W_ens.shape)
    
    #mag_perturb = gradient_magnitude(W_ens, g_ema, method='perturb')
    
    
    
    #print(mag_perturb.norm())
    
    colors_mix = ['bo-', 'ro-', 'ko-', 'go-', 'yo-']
    colors_pert = ['b--', 'r--', 'k--', 'g--', 'y--']
    
    mix_norms_u = []
    mix_norms_v = []
    mix_norms_t2m = []
    
    for nmix in range(12) :
        mag_mix0 = gradient_magnitude(W_ens, g_ema, method='strong_mixing', n_mix = nmix)
        print(torch.sqrt((mag_mix0**2).sum(dim = (1,2,3))).mean(0))
        mix_norms_u.append(torch.sqrt((mag_mix0[:,0:1,:,:]**2).sum(dim = (1,2,3))).mean(0))
        mix_norms_v.append(torch.sqrt((mag_mix0[:,1:2,:,:]**2).sum(dim = (1,2,3))).mean(0))
        mix_norms_t2m.append(torch.sqrt((mag_mix0[:,2:,:,:]**2).sum(dim = (1,2,3))).mean(0))
    plt.plot(mix_norms_u, colors_mix[0], linewidth=2,label = 'concat u' )
    plt.plot(mix_norms_v, colors_mix[1], linewidth=2,label = 'concat v' )
    plt.plot(mix_norms_t2m, colors_mix[2], linewidth=2,label = 'concat t2m' )
    
    mix_th_norms_u = []
    mix_th_norms_v = []
    mix_th_norms_t2m = []
    
    for nmix in range(12) :
        mag_mix0 = gradient_magnitude(W_ens, g_ema, method='thresh_mix', n_mix = nmix)
        print(torch.sqrt((mag_mix0**2).sum(dim = (1,2,3))).mean(0))
        mix_th_norms_u.append(torch.sqrt((mag_mix0[:,0:1,:,:]**2).sum(dim = (1,2,3))).mean(0))
        mix_th_norms_v.append(torch.sqrt((mag_mix0[:,1:2,:,:]**2).sum(dim = (1,2,3))).mean(0))
        mix_th_norms_t2m.append(torch.sqrt((mag_mix0[:,2:,:,:]**2).sum(dim = (1,2,3))).mean(0))
    plt.plot(mix_th_norms_u, colors_pert[0], linewidth=2,label = 'threshold u' )
    plt.plot(mix_th_norms_v, colors_pert[1], linewidth=2,label = 'threshold v' )
    plt.plot(mix_th_norms_t2m, colors_pert[2], linewidth=2,label = 'threshold t2m' )
    
    
    plt.yscale('log')
    plt.xlabel(r'$n_{mix}$')
    plt.ylabel(r'$\Delta G$')
    plt.title(r'Evolution of $\Delta G = G(W) -G(W + \epsilon_{n_{mix}}) $')
    plt.grid()
    plt.legend()
    plt.savefig(Path_ckpt_encoder + 'threshold_vs_cat_pervar.png')
    plt.close()
    
    """
    
    for i, maglog in enumerate(range(-2,1)) :
        
        print(maglog)
        
        mag = 10**(maglog)
    
        pert_norms = []
        
        lab = r'$\sigma={}$'.format(str(mag))
        for nmix in range(12) :
            mag_mix = gradient_magnitude(W_ens, g_ema, magn = mag, method='mixing', n_mix = nmix)
            print(torch.sqrt((mag_mix**2).sum(dim = (1,2,3))).mean(0))
        
            pert_norms.append(torch.sqrt((mag_mix**2).sum(dim = (1,2,3))).mean(0))
            
        
        plt.plot(pert_norms, colors_pert[i], linewidth=2, label = lab)
    
    for i, maglog in enumerate(range(-2,1)) :
        
        print(maglog)
        
        mag = 10**(maglog)
    
        mix_norms = []
        for nmix in range(12) :
            mag_mix = gradient_magnitude(W_ens, g_ema, magn = mag, method='mixing_lone', n_mix=nmix)
            print(torch.sqrt((mag_mix**2).sum(dim = (1,2,3))).mean(0))
        
            mix_norms.append(torch.sqrt((mag_mix**2).sum(dim = (1,2,3))).mean(0))
        plt.plot(mix_norms, colors[i], linewidth=2)
    plt.savefig(Path_ckpt_encoder + 'mix_norms_test_lone_{}.png'.format(str(mag)))
    plt.close()
    """
    
    
    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 14:53:25 2023

@author: brochetc


tests for gradient magnitude verification

"""

import torch


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def style_mixing_ensemble(Ensemble_w, g_ema, n_split, generate=False):
    """
    
    mixes random styles above layer n_split in stylegan2 architecture
    with the styles of ensembles
    
    """
    
    g_ema.eval()
    assert n_split <= g_ema.num_layers + 1
    
    n_ensemble = Ensemble_w.shape[0]

    
    
    ### making random latents
    noise_sample = torch.randn(n_ensemble, g_ema.style_dim, device=device)
    latent_out = g_ema.style(noise_sample).unsqueeze(1)
    w_l_o = latent_out.repeat(1,12,1)
    
    ## mixing

    w_mix = torch.cat((Ensemble_w[:,:n_split,:], w_l_o[:,n_split:,:]), dim = 1).float()
    
    if generate :
        
         w_mix.to(device)
         mixed_sample, _ = g_ema([w_mix], input_is_latent=True)
         return mixed_sample
     
    return w_mix

def style_mixing_threshold(Ensemble_w, threshold, g_ema, generate=False):
    """
    
    mixes random styles with the threshold (might be a tensor that requires_grad)
    
    threshold must be of size N_ens x N_latent
    
    """
    
    g_ema.eval()
    
    n_ensemble = Ensemble_w.shape[0]
   
    w = Ensemble_w.to(device)
    
    ### making random latents
    noise_sample = torch.randn(n_ensemble, g_ema.style_dim, device=device)
    latent_out = g_ema.style(noise_sample).unsqueeze(1)
    w_l_o = latent_out.repeat(1,g_ema.n_latent,1).to(device)
    
    ## mixing 

    threshold = threshold.view(n_ensemble, g_ema.n_latent , g_ema.style_dim)
        
    w_mix = w * threshold + w_l_o * (1.0 - threshold)
    
    if generate :
        
         mixed_sample, _ = g_ema([w_mix], input_is_latent=True)
         return mixed_sample
     
    return w_mix

def gradient_magnitude(w, g_ema,  method='perturb', magn = 0.01, n_mix = 6):
    
    """"computing 'gradient magnitude' with two different methods
    
    --> 'perturb' : takes the SAME w (repeated n_styles times)
    
    """
    
    assert method in ['perturb_mix','mixing', 'strong_mixing','thresh_mix']
    
    
    diff = torch.empty((16*10,3,128,128))
    
    if method=='perturb_mix' :
        
        noise = torch.empty((16*10, w.shape[1] - n_mix, 512)).normal_(0.0, magn)
        
        
        with torch.no_grad():
            
            w0 = w[:,0:1,:]
            
            w0 = w0.repeat(1,12,1)
            
            w_cuda = w0.to(device)
            
            samples,_ = g_ema([w_cuda], input_is_latent=True)
            
            samples = samples.detach().cpu()
            

    
        for i in range(10) : 
                        
            pert = noise[16 * i : 16 * (i+1)].to(device)
                        
            w_test = torch.cat((w0[:,:n_mix,:], w0[:, n_mix:, :] + pert), dim = 1)
                    
            w_test.requires_grad_()
            
            diff[16 * i : 16 * (i+1)] = g_ema([w_test], input_is_latent=True)[0].detach().cpu() - samples
        
        return diff
    
    if method=='mixing' :
        
        noise = torch.empty((16*10, w.shape[1] - n_mix, w.shape[2])).normal_(0.0, magn)
        
        with torch.no_grad():
        
            w_cuda = w.to(device)
            
            samples = g_ema([w_cuda], input_is_latent=True)[0].detach().cpu()
    
        for i in range(10) : 
            
            pert = noise[16 * i : 16 * (i+1)].to(device)
        
            w_test = torch.cat((w[:,:n_mix,:], w[:, n_mix:, :] + pert), dim = 1)

            with torch.no_grad() :
                diff[16 * i : 16 * (i+1)] = g_ema([w_test], input_is_latent=True)[0].detach().cpu() - samples
        
        return diff
    
    if method=='strong_mixing' :
        
        with torch.no_grad():
        
            w_cuda = w.to(device)
            
            samples = g_ema([w_cuda], input_is_latent=True)[0].detach().cpu()
    
        for i in range(10) : 
            
            w_mix = style_mixing_ensemble(w, g_ema, n_mix)

            with torch.no_grad() :
                diff[16 * i : 16 * (i+1)] = g_ema([w_mix], input_is_latent=True)[0].detach().cpu() - samples
        
        return diff
    
    if method=='thresh_mix' :
        
        with torch.no_grad():
        
            w_cuda = w.to(device)
            
            samples = g_ema([w_cuda], input_is_latent=True)[0].detach().cpu()
            
        threshold = torch.ones_like(w)
        
        threshold[:,n_mix:,:] = torch.zeros((w.shape[0], w.shape[1]-n_mix, w.shape[2]))
        
        for i in range(10) : 
            
            w_mix = style_mixing_threshold(w, threshold, g_ema,)

            with torch.no_grad() :
                diff[16 * i : 16 * (i+1)] = g_ema([w_mix], input_is_latent=True)[0].detach().cpu() - samples
        
        return diff
        
        
        
        
        
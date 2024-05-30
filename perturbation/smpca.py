#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:04:50 2023

@author: brochetc

Syle-Mixed PCA
"""
import torch
import numpy as np
import perturbation.pca_stylegan as pca
from math import ceil
import random
import torch.nn.functional as F

def sample_sigma_cut_w(Cov, w_avg, N_samples,Whitening,w0,G, scale=1.0, device='cuda:0', verbose=False):
    """
    Linear analysis, but samples are drawn from the z -> w mapping. PCA is conducted on the space of whitened features.
    We thus need a Whitening matrix (shape 512 x 512) and Coloring matrix (shape 512 x 512) to go from W feature space to whitened and back
    Coloring and Whitening matrices will be the inverse of each other; and the same for each style
    w0 : general mean of the W space
    """
    R, D = w_avg.shape # Repeats, Dimension (typicallly = 14, 512)
    #w_avg = Ens_w.mean(dim=0)
    z = torch.empty((N_samples,D)).normal_().contiguous().to(device)
    with torch.no_grad():
        w = G.style(z)
        w  = w #* torch.rsqrt(torch.linalg.norm(w, dim=-1, keepdims=True) + 1e-8)
    diff = torch.bmm(Whitening.to(device).unsqueeze(0).repeat(N_samples,1,1) , (w - w.mean(dim=0)).unsqueeze(-1)) # diff of shape N_samples  x D
    if verbose:
        print("diff shape", diff.shape)
    
    new_w = torch.einsum('abc, dc-> dab',Cov, diff.squeeze(dim=-1))
    if verbose:
        print("new_w perturbation shape",new_w.shape)
    assert torch.isfinite(new_w).all() #stop if compute is instable
    
    if verbose:
        print("scale", scale)
        
    res = w_avg + scale * new_w.view(N_samples, R, D) #* torch.rsqrt(new_w.view(N_samples,R,D).var(dim=0, unbiased=True).mean(dim=-1, keepdims=True) + 1e-8)

    if verbose: 
        print("res shape and diff to mean ",res.shape, 
            torch.linalg.norm((res - w_avg).mean(dim=0)),
            torch.max(torch.abs((res-w_avg).mean(dim=0))))

    return res 

def sm_pca(
        Ens_w, 
        G, 
        N_samples, 
        sm_ind = [0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
        device = 'cuda:0', 
        sample_rule = 'stochastic',
        N_seeds=16, 
        random_unbias=False,
        scale=1.0,
        interp=0.0,
        verbose=False,
        Whitening=None,
        Coloring=None,
        w0=None,
        renorm=False
    ):
    
    N, R, D = Ens_w.shape
    per_cond = int(ceil(N_samples / N_seeds))
    
    Ens_final = np.zeros((N * per_cond,3,256,256), dtype = 'float32')
    w_final = np.zeros((N * per_cond, R, D))
    
    sm_ind_np = np.array(sm_ind).astype(np.bool_)

    w_extract = Ens_w[:,sm_ind_np,:].to(device)
    if verbose : 
        print(f"Extracted w_extract {w_extract.size()}")
    n_styles_pert = w_extract.size()[1]

    if sample_rule=='stochastic':
        assert Coloring is not None
        assert Whitening is not None
        assert w0 is not None
        if verbose: print(f"scale {scale}")
        if n_styles_pert>0:
            Cov, w_avg = pca.computeReducedCovarianceW(w_extract,cut=N-1,
                                                verbose=verbose,renorm=renorm)
        else:
            Cov, w_avg = None, None
    else:
        Cov, w_avg = None,None

    if sample_rule=='stochastic':
        Ens_w1 = Ens_w.to(device)
        if N_seeds < N:
            seeds = random.sample(range(N), N_seeds)
            Ens_w1 = Ens_w[seeds].to(device)
            print(Ens_w1.shape)
        
    with torch.no_grad():
        for k in range(N_seeds) : # generating a common multiple of each conditioning sample
            if verbose: print(f"member {k} is fixed")
            if sample_rule=='stochastic':
                if n_styles_pert:
                    z = torch.empty((per_cond,D)).normal_().contiguous().to(device)
                    with torch.no_grad():
                        w = G.style(z)
                    diff = torch.bmm(Whitening.to(device).unsqueeze(0).repeat(per_cond,1,1), 
                                     (w - w.mean(dim=0)).unsqueeze(-1)) # diff of shape N_samples  x D
                    new_w = torch.einsum('abc, dc-> dab',Cov, diff.squeeze(dim=-1))

                w_start = interp.view(1,14,1) * Ens_w1.mean(dim=0) + (1.0 - interp).view(1,14,1) * Ens_w1[k]
                if (R - n_styles_pert)>0:
                    z = torch.empty((per_cond,512)).normal_().to(device)
                    with torch.no_grad():
                        w_nopca = G.style(z)
                    if n_styles_pert>0:
                        w_pert = torch.cat([new_w, (w_nopca - w_nopca.mean(dim=0)).unsqueeze(1).repeat(1,(R-n_styles_pert),1)],dim=1)
                    else:
                        w_pert = (w_nopca - w_nopca.mean(dim=0)).unsqueeze(1).repeat(1,(R-n_styles_pert),1)
                else:
                    w_pert = new_w
                w_new = w_start + scale.view(1,14,1) * w_pert
                
            elif sample_rule == 'extrapolation' :
                w_interm = []
                for kk in range(k, N_seeds) :
                    if k != kk:
                        print(k,kk)
                        w_interm.append(( Ens_w[k] + 1.5 * (Ens_w[kk] - Ens_w[k])).to(device))
                        w_interm.append(( Ens_w[kk] + 1.5 * (Ens_w[k] - Ens_w[kk])).to(device))

                if k==(N_seeds-1) :

                    return Ens_final[:N_samples], w_final

                else :

                    w_new = torch.stack(w_interm, axis=0)
                    per_cond = w_new.size()[0]
            else:
                raise ValueError(f"sampling rule not good, should be 'stochastic' or 'extrapolation' but is {sample_rule}")

            assert torch.isfinite(w_new).all()
            if verbose : print('wnew', w_new.shape)
            w = w_new
            sample, _, _  = G([w.to(device)],input_is_latent = True)
            Ens_final[k * per_cond : (k + 1) * per_cond] = sample.detach().cpu().numpy()
            w_final[k * per_cond : (k + 1) * per_cond] = w.detach().cpu().numpy()
    
    return Ens_final[:N_samples], w_final

def fast_style_mixing(interp, scale, batch_w, Cov, w_avg, w0, n_samples, G, Whitening, device='cpu', scale_rule='linear'):
    """
    Perform style mixing using interpolation coefficients (alpha's) and scale coefficients (beta's)
    and make the resulting physical samples differentiable wrt alpha's and beta's
    To be used with scale_tune script
    """
    R, D = w_avg.shape # Repeats, Dimension (typicallly = 14, 512)
    n_styles_no_pca = 14 - R
    w_start = F.sigmoid(interp).view(1,14,1) * batch_w.mean(dim=0) + (1.0 - F.sigmoid(interp).view(1,14,1)) * batch_w
    
    # perturbation on scales implying filtering
    if R>0:
        z = torch.empty((n_samples,D)).normal_().contiguous().to(device)
        with torch.no_grad():
            w = G.style(z)
        diff = torch.bmm(Whitening.to(device).unsqueeze(0).repeat(n_samples,1,1) , (w - w.mean(dim=0)).unsqueeze(-1)) # diff of shape N_samples  x D
        new_w = torch.einsum('abc, dc-> dab',Cov, diff.squeeze(dim=-1))

    # perturbation on scales implying random noise
    if n_styles_no_pca>0:
        z = torch.empty((n_samples,512)).normal_().to(device)
        with torch.no_grad():
            w_nopca = G.style(z)
        if R>0:
            w_pert = torch.cat([new_w, (w_nopca-w_nopca.mean(dim=0)).unsqueeze(1).repeat(1,n_styles_no_pca,1)],dim=1)
        else:
            w_pert = (w_nopca-w_nopca.mean(dim=0)).unsqueeze(1).repeat(1,n_styles_no_pca,1)
    else:
        w_pert = new_w
    
    # scales viewed as linear parameters
    if scale_rule=='linear':
        res = w_start + scale.view(1,14,1) * w_pert
        gen,_,_ = G([res], input_is_latent=True)

    # constraining scales to be strictly in (0,1)
    elif scale_rule=='sigmoid':
        res = w_start + F.sigmoid(scale).view(1,14,1) * w_pert
    gen,_,_ = G([res], input_is_latent=True)
    
    return gen
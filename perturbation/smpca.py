#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:04:50 2023

@author: brochetc

Style-Mixed PCA
"""
import random
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
import perturbation.pca_stylegan as pca


def sm_pca(
        Ens_w, 
        G, 
        N_samples, 
        sm_ind = [0]*14, 
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
        renorm=False,
        import_perturbation=False,
        save_perturbation=False,
        path_perturbation=''
    ):
    
    N, R, D = Ens_w.shape
    N_seeds = min(N_seeds, N)
    per_cond = int(ceil(N_samples / N_seeds))

    # --- replace pre-allocation with lists ---
    Ens_perturbated_list = []
    w_perturbated_list = []
    perturbation_list = [] if save_perturbation and not import_perturbation else None

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
            Cov, w_avg = pca.compute_K_covariance(
                                                        w_extract,
                                                        cut=N-1,
                                                        verbose=verbose,
                                                        renorm=renorm
            )
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
        for k in range(N_seeds):
            if verbose: print(f"member {k} is fixed")
            if sample_rule=='stochastic':
                #w_start = interp.view(1,R,1) * Ens_w1.mean(dim=0) + (1.0 - interp).view(1,R,1) * Ens_w1[k]
                w_start = interp * Ens_w1.mean(dim=0) + (1.0 - interp) * Ens_w1[k]

                if n_styles_pert:
                    z = torch.empty((per_cond,D)).normal_().contiguous().to(device)
                    with torch.no_grad():
                        w = G.style(z)
                    diff = torch.bmm(
                        Whitening.to(device).unsqueeze(0).repeat(per_cond,1,1), 
                        (w - w.mean(dim=0)).unsqueeze(-1)
                    )
                    new_w = torch.einsum('abc, dc-> dab',Cov, diff.squeeze(dim=-1))

                if import_perturbation :
                    if path_perturbation :
                        w_pert = np.load(path_perturbation)[k * per_cond : (k + 1) * per_cond]
                        w_pert = torch.from_numpy(w_pert).to(device)
                    else :
                        raise FileNotFoundError('Specify a path for the perturbation')
                else :
                    if (R - n_styles_pert) > 0:
                        z = torch.empty((per_cond,512)).normal_().to(device)
                        with torch.no_grad():
                            w_nopca = G.style(z)
                        if n_styles_pert > 0:
                            w_pert = torch.cat([new_w, (w_nopca - w_nopca.mean(dim=0)).unsqueeze(1).repeat(1,(R-n_styles_pert),1)],dim=1)
                        else:
                            w_pert = (w_nopca - w_nopca.mean(dim=0)).unsqueeze(1).repeat(1,(R-n_styles_pert),1)
                    else:
                        w_pert = new_w

                if import_perturbation:
                    w_new = w_start + w_pert.type(w_start.type())
                else:
                    #w_new = w_start + scale.view(1,R,1) * w_pert
                    w_new = w_start + scale * w_pert
                
            elif sample_rule == 'extrapolation':
                if save_perturbation or import_perturbation:
                    raise NotImplementedError
                w_interm = []
                for kk in range(k, N_seeds):
                    if k != kk:
                        w_interm.append((Ens_w[k] + 1.5 * (Ens_w[kk] - Ens_w[k])).to(device))
                        w_interm.append((Ens_w[kk] + 1.5 * (Ens_w[k] - Ens_w[kk])).to(device))
                if k==(N_seeds-1):
                    break
                else:
                    w_new = torch.stack(w_interm, axis=0)
            else:
                raise ValueError(f"sampling rule not good, should be 'stochastic' or 'extrapolation'")

            assert torch.isfinite(w_new).all()
            if verbose: print('wnew', w_new.shape)

            sample, _, _ = G([w_new.to(device)], input_is_latent=True)

            # --- append to lists ---
            Ens_perturbated_list.append(sample.detach().cpu().numpy())
            w_perturbated_list.append(w_new.detach().cpu().numpy())
            if perturbation_list is not None:
                perturbation_list.append((scale.view(1,R,1) * w_pert).detach().cpu().numpy())

    # --- concatenate all at the end ---
    Ens_perturbated = np.concatenate(Ens_perturbated_list, axis=0)[:N_samples]
    w_perturbated = np.concatenate(w_perturbated_list, axis=0)
    if perturbation_list is not None:
        perturbation = np.concatenate(perturbation_list, axis=0)
        return Ens_perturbated, (w_perturbated, perturbation)
    else:
        return Ens_perturbated, w_perturbated


def fast_style_mixing(
    alphas,
    betas,
    batch_w,
    K,
    w_avg,
    n_samples,
    G,
    Whitening,
    device="cpu",
    beta_rule="linear",
):
    """
    Perform style mixing using interpolation coefficients (alpha's) and scale coefficients (beta's)
    and make the resulting physical samples differentiable wrt alpha's and beta's
    To be used with scale_tune script
    """
    R, D = w_avg.shape  # Repeats, Dimension (typically = n_layers, 512)
    B, n_layers, w_dim = batch_w.shape

    # --- build w_start ---
    interp_ = F.sigmoid(alphas).view(1, n_layers, 1).expand(B, n_layers, 1)
    w_mean = batch_w.mean(dim=0, keepdim=True).expand_as(batch_w)  # (B, n_layers, w_dim)
    w_start = interp_ * w_mean + (1.0 - interp_) * batch_w         # (B, n_layers, w_dim)

    # how many layers are *not* covered by PCA components
    n_styles_no_pca = n_layers - R

    # perturbation on scales implying filtering
    if R > 0:
        z = torch.empty((n_samples, D)).normal_().contiguous().to(device)
        with torch.no_grad():
            w = G.style(z)
        diff = torch.bmm(
            Whitening.to(device).unsqueeze(0).repeat(n_samples, 1, 1),
            (w - w.mean(dim=0)).unsqueeze(-1)
        )
        new_w = torch.einsum('abc, dc->dab', K, diff.squeeze(dim=-1))  # (n_samples, R, D)

    # perturbation on scales implying random noise
    if n_styles_no_pca > 0:
        z = torch.empty((n_samples, 512)).normal_().to(device)
        with torch.no_grad():
            w_nopca = G.style(z)
        rand_block = (w_nopca - w_nopca.mean(dim=0)).unsqueeze(1).repeat(1, n_styles_no_pca, 1)
        if R > 0:
            w_pert = torch.cat([new_w, rand_block], dim=1)
        else:
            w_pert = rand_block
    else:
        w_pert = new_w

    # match batch size if needed
    if w_pert.shape[0] != B:
        if w_pert.shape[0] == 1:
            w_pert = w_pert.expand(B, -1, -1)
        else:
            w_pert = w_pert[:B]

    # --- apply beta scaling ---
    if beta_rule == "linear":
        scale_eff = betas.view(1, n_layers, 1).expand_as(w_pert)
        res = w_start + scale_eff * w_pert

    elif beta_rule == "sigmoid":
        scale_eff = F.sigmoid(betas).view(1, n_layers, 1).expand_as(w_pert)
        res = w_start + scale_eff * w_pert

    gen, _, _ = G([res], input_is_latent=True)
    return gen


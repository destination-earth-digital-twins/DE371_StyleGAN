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
import perturbation.pca_stylegan as pca


def sm_pca(
    Ens_w,
    G,
    N_samples,
    sm_ind=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    device="cuda:0",
    sample_rule="stochastic",
    N_seeds=16,
    betas=1.0,
    alphas=0.0,
    verbose=False,
    Whitening=None,
    Coloring=None,
    w0=None,
    renorm=False,
    import_perturbation=False,
    save_perturbation=False,
    path_perturbation='',
    Ens_feature=None,
    feature_id=6,
    feature_scale=1,
    temporal_consistency=False,
    dt=3,
    theta=0.5,
    sigma=0.1,
    current_timestep=0,
    temporal_noises=[]

):

    N, R, D = Ens_w.shape
    per_cond = int(ceil(N_samples / N_seeds))

    Ens_final = np.zeros((N * per_cond, 3, 256, 256), dtype="float32")
    w_final = np.zeros((N * per_cond, R, D))

    if Ens_feature is not None:
        Ens_feature = Ens_feature.to(device)
        noise = G.make_noise()

    if save_perturbation and not import_perturbation :
        perturbation = np.zeros((N * per_cond, R, D))
    if save_perturbation and import_perturbation :
        raise NotImplementedError
    
    sm_ind_np = np.array(sm_ind).astype(np.bool_)

    w_extract = Ens_w[:, sm_ind_np, :].to(device)
    if verbose:
        print(f"Extracted w_extract {w_extract.size()}")
    n_styles_pert = w_extract.size()[1]

    if sample_rule == "stochastic":
        assert Coloring is not None
        assert Whitening is not None
        assert w0 is not None
        if verbose:
            print(f"betas (scale factor) {betas}")
        if n_styles_pert > 0:
            K, _ = pca.compute_K_covariance(
                w_extract, cut=N - 1, verbose=verbose, renorm=renorm
            )
        else:
            K = None
    else:
        K = None

    if sample_rule == "stochastic":
        Ens_w1 = Ens_w.to(device)
        if N_seeds < N:
            seeds = random.sample(range(N), N_seeds)
            Ens_w1 = Ens_w[seeds].to(device)
            print(Ens_w1.shape)

    with torch.no_grad():
        
        for k in range(
            N_seeds
        ):  # generating a common multiple of each conditioning sample
            if verbose:
                print(f"member {k} is fixed")
            if sample_rule == "stochastic":
                if n_styles_pert:
                    z = torch.empty((per_cond, D)).normal_().contiguous().to(device)
                    with torch.no_grad():
                        w = G.style(z)
                    diff = torch.bmm(
                        Whitening.to(device).unsqueeze(0).repeat(per_cond, 1, 1),
                        (w - w.mean(dim=0)).unsqueeze(-1),
                    )  # diff of shape N_samples  x D
                    new_w = torch.einsum("abc, dc-> dab", K, diff.squeeze(dim=-1))

                w_start = (
                    alphas.view(1, 14, 1) * Ens_w1.mean(dim=0)
                    + (1.0 - alphas).view(1, 14, 1) * Ens_w1[k]
                )
                if import_perturbation and path_perturbation is not None:
                    w_pert = torch.tensor(np.load(path_perturbation)[k * per_cond : (k + 1) * per_cond].astype(np.float32)).to(device)
                else:
                    if not temporal_consistency :
                        if (R - n_styles_pert) > 0:
                            z = torch.empty((per_cond, 512)).normal_().to(device)
                            with torch.no_grad():
                                w_nopca = G.style(z)
                            if n_styles_pert > 0:
                                w_pert = torch.cat(
                                    [
                                        new_w,
                                        (w_nopca - w_nopca.mean(dim=0))
                                        .unsqueeze(1)
                                        .repeat(1, (R - n_styles_pert), 1),
                                    ],
                                    dim=1,
                                )
                            else:
                                w_pert = (
                                    (w_nopca - w_nopca.mean(dim=0))
                                    .unsqueeze(1)
                                    .repeat(1, (R - n_styles_pert), 1)
                                )
                        else:
                            w_pert = new_w
                    else :
                        if path_perturbation is None:
                            raise ImportError(f'path_perturbation parameter has to be imported but instead got : {path_perturbation}')
                        w_pert_init = torch.tensor(np.load(path_perturbation)[k * per_cond : (k + 1) * per_cond].astype(np.float32)).to(device)
                        if current_timestep == 0:
                            w_pert = w_pert_init.detach().clone()
                        else :
                            w_pert = w_pert_init.detach().clone() * (1-theta*dt)**(current_timestep)
                            list_temporal_noise = [temporal_noises[current_timestep-k]*(1-theta*dt)**k for k in range(current_timestep)]
                            w_pert += sigma * torch.sqrt(torch.tensor(dt)) * torch.from_numpy(np.array(list_temporal_noise)).sum()

      
                w_new = w_start + betas.view(1, 14, 1) * w_pert

            elif sample_rule == "extrapolation":
                if save_perturbation or import_perturbation:
                    raise NotImplementedError
                w_interm = []
                for kk in range(k, N_seeds):
                    if k != kk:
                        print(k, kk)
                        w_interm.append(
                            (Ens_w[k] + 1.5 * (Ens_w[kk] - Ens_w[k])).to(device)
                        )
                        w_interm.append(
                            (Ens_w[kk] + 1.5 * (Ens_w[k] - Ens_w[kk])).to(device)
                        )

                if k == (N_seeds - 1):

                    return Ens_final[:N_samples], w_final

                else:

                    w_new = torch.stack(w_interm, axis=0)
                    per_cond = w_new.size()[0]
            else:
                raise ValueError(
                    f"sampling rule unknown, should be 'stochastic' or 'extrapolation' but is {sample_rule}"
                )

            assert torch.isfinite(w_new).all()
            if verbose:
                print("wnew", w_new.shape)
            w = w_new

            # features for generator
            features_in = None
            if Ens_feature is not None:
                
                print('shape w_inv',Ens_w1[k].unsqueeze(0).shape)
                sample, features_out_inv, _ = G([(Ens_w1[k].unsqueeze(0)).to(device)], input_is_latent=True, return_features=True, noise=noise)

                print('shape features_out_inv',features_out_inv[feature_id].shape)
                print('shape w_new',w.shape)
                sample, features_out_pert, _ = G([w.to(device)], input_is_latent=True, return_features=True, noise=noise)
                print('shape features_out_pert',features_out_pert[feature_id].shape)

                print('shape features from encoder',Ens_feature[k].shape)
                F = Ens_feature[k].unsqueeze(0).repeat(per_cond, 1, 1, 1)
                print('shape features from encoder ready',F.shape)
                feature_map_from_pert = features_out_pert[feature_id]
                feature_map_from_inv = features_out_inv[feature_id].repeat(per_cond, 1, 1, 1)
                
                feature_to_insert = F + feature_map_from_pert - feature_map_from_inv
                features_in = [None]*(feature_id)+ [feature_to_insert] + [None]*(13-(feature_id))
                sample, _, _ = G([w.to(device)],  features_in=features_in, feature_scale=feature_scale, input_is_latent=True, noise=noise)
            else :
                
                sample, _, _ = G([w.to(device)],  input_is_latent=True)

            Ens_final[k * per_cond : (k + 1) * per_cond] = sample.detach().cpu().numpy()
            w_final[k * per_cond : (k + 1) * per_cond] = w.detach().cpu().numpy()
            if save_perturbation and not import_perturbation:
                perturbation[k * per_cond : (k + 1) * per_cond] = (w_pert).detach().cpu().numpy()

    if save_perturbation and not import_perturbation :
        return Ens_final[:N_samples], (w_final, perturbation)
    else :
        return Ens_final[:N_samples], w_final


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
    R, D = w_avg.shape  # Repeats, Dimension (typicallly = 14, 512)
    n_styles_no_pca = 14 - R
    
    # sigmoid is applied to alphas --> stored value is thus sigmoid(alphas)
    w_start = (
        F.sigmoid(alphas).view(1, 14, 1) * batch_w.mean(dim=0)
        + (1.0 - F.sigmoid(alphas).view(1, 14, 1)) * batch_w
    )

    # perturbation on styles implying filtering
    if R > 0:
        z = torch.empty((n_samples, D)).normal_().contiguous().to(device)
        with torch.no_grad():
            w = G.style(z)
        diff = torch.bmm(
            Whitening.to(device).unsqueeze(0).repeat(n_samples, 1, 1),
            (w - w.mean(dim=0)).unsqueeze(-1),
        )  # diff of shape N_samples  x D
        new_w = torch.einsum("abc, dc-> dab", K, diff.squeeze(dim=-1))

    # perturbation on styles implying random noise
    if n_styles_no_pca > 0:
        z = torch.empty((n_samples, 512)).normal_().to(device)
        with torch.no_grad():
            w_nopca = G.style(z)
        if R > 0:
            w_pert = torch.cat(
                [
                    new_w,
                    (w_nopca - w_nopca.mean(dim=0))
                    .unsqueeze(1)
                    .repeat(1, n_styles_no_pca, 1),
                ],
                dim=1,
            )
        else:
            w_pert = (
                (w_nopca - w_nopca.mean(dim=0))
                .unsqueeze(1)
                .repeat(1, n_styles_no_pca, 1)
            )
    else:
        w_pert = new_w

    # betas viewed as linear parameters
    if beta_rule == "linear":
        res = w_start + betas.view(1, 14, 1) * w_pert
        gen, _, _ = G([res], input_is_latent=True)

    # constraining betas to be strictly in (0,1)
    elif beta_rule == "sigmoid":
        res = w_start + F.sigmoid(betas).view(1, 14, 1) * w_pert
    gen, _, _ = G([res], input_is_latent=True)

    return gen


def fast_style_mixing_temporal(
    dt,
    theta,
    gamma,
    batch_w,
    batch_w_next,
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
    R, D = w_avg.shape  # Repeats, Dimension (typicallly = 14, 512)
    n_styles_no_pca = 14 - R
    
    # sigmoid is applied to theta --> stored value is thus sigmoid(theta)
    w_start = (
        F.sigmoid(theta).view(1, 14, 1) * dt * batch_w_next
        + (1.0 - F.sigmoid(theta).view(1, 14, 1) * dt) * batch_w
    )

    # perturbation on styles implying filtering
    if R > 0:
        z = torch.empty((n_samples, D)).normal_().contiguous().to(device)
        with torch.no_grad():
            w = G.style(z)
        diff = torch.bmm(
            Whitening.to(device).unsqueeze(0).repeat(n_samples, 1, 1),
            (w - w.mean(dim=0)).unsqueeze(-1),
        )  # diff of shape N_samples  x D
        new_w = torch.einsum("abc, dc-> dab", K, diff.squeeze(dim=-1))

    # perturbation on styles implying random noise
    if n_styles_no_pca > 0:
        z = torch.empty((n_samples, 512)).normal_().to(device)
        with torch.no_grad():
            w_nopca = G.style(z)
        if R > 0:
            w_pert = torch.cat(
                [
                    new_w,
                    (w_nopca - w_nopca.mean(dim=0))
                    .unsqueeze(1)
                    .repeat(1, n_styles_no_pca, 1),
                ],
                dim=1,
            )
        else:
            w_pert = (
                (w_nopca - w_nopca.mean(dim=0))
                .unsqueeze(1)
                .repeat(1, n_styles_no_pca, 1)
            )
    else:
        w_pert = new_w
    # print(w_pert.shape)
    # gamma viewed as linear parameters
    if beta_rule == "linear":
        res = w_start + torch.mul(gamma.view(1, 14, 1) * w_pert, torch.sqrt(torch.tensor(dt)))
        gen, _, _ = G([res], input_is_latent=True)

    # constraining gamma to be strictly in (0,1)
    elif beta_rule == "sigmoid":
        res = w_start + torch.mul(F.sigmoid(gamma).view(1, 14, 1) * w_pert , torch.sqrt(torch.tensor(dt)))
    gen, _, _ = G([res], input_is_latent=True)

    return gen


def sm_pca_temporal(
        Ens_w,
        Ens_w_next,
        G,
        N_samples,
        sm_ind=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        device="cuda:0",
        sample_rule="stochastic",
        N_seeds=16,
        dt=3,
        thetas=1.0,
        gammas=0.0,
        verbose=False,
        Whitening=None,
        Coloring=None,
        w0=None,
        renorm=False,
    ):

        N, R, D = Ens_w.shape
        per_cond = int(ceil(N_samples / N_seeds))

        Ens_final = np.zeros((N * per_cond, 3, 256, 256), dtype="float32")
        w_final = np.zeros((N * per_cond, R, D))
        
        sm_ind_np = np.array(sm_ind).astype(np.bool_)

        w_extract = Ens_w[:, sm_ind_np, :].to(device)
        w_extract_next = Ens_w_next[:, sm_ind_np, :].to(device)
        delta_w_extract = w_extract_next - w_extract

        if verbose:
            print(f"Extracted w_extract {w_extract.size()}")
        n_styles_pert = w_extract.size()[1]

        if sample_rule == "stochastic":
            assert Coloring is not None
            assert Whitening is not None
            assert w0 is not None
            if verbose:
                print(f"gammas (scale factor) {gammas}")
            if n_styles_pert > 0:
                K, _ = pca.compute_K_covariance(
                    delta_w_extract, cut=N - 1, verbose=verbose, renorm=renorm
                )
            else:
                K = None
        else:
            K = None

        if sample_rule == "stochastic":
            Ens_w1 = Ens_w.to(device)
            Ens_w1_next = Ens_w_next.to(device)
            if N_seeds < N:
                seeds = random.sample(range(N), N_seeds)
                Ens_w1 = Ens_w[seeds].to(device)
                Ens_w1_next = Ens_w_next[seeds].to(device)
                print(Ens_w1.shape)

        with torch.no_grad():
            
            for k in range(
                N_seeds
            ):  # generating a common multiple of each conditioning sample
                if verbose:
                    print(f"member {k} is fixed")
                if sample_rule == "stochastic":
                    if n_styles_pert:
                        z = torch.empty((per_cond, D)).normal_().contiguous().to(device)
                        with torch.no_grad():
                            w = G.style(z)
                        diff = torch.bmm(
                            Whitening.to(device).unsqueeze(0).repeat(per_cond, 1, 1),
                            (w - w.mean(dim=0)).unsqueeze(-1),
                        )  # diff of shape N_samples  x D
                        new_w = torch.einsum("abc, dc-> dab", K, diff.squeeze(dim=-1))

                    w_start = (
                        thetas.view(1, 14, 1) * dt * Ens_w1_next[k]
                        + (1.0 - thetas.view(1, 14, 1) * dt) * Ens_w1[k]
                    )

                    if (R - n_styles_pert) > 0:
                        z = torch.empty((per_cond, 512)).normal_().to(device)
                        with torch.no_grad():
                            w_nopca = G.style(z)
                        if n_styles_pert > 0:
                            w_pert = torch.cat(
                                [
                                    new_w,
                                    (w_nopca - w_nopca.mean(dim=0))
                                    .unsqueeze(1)
                                    .repeat(1, (R - n_styles_pert), 1),
                                ],
                                dim=1,
                            )
                        else:
                            w_pert = (
                                (w_nopca - w_nopca.mean(dim=0))
                                .unsqueeze(1)
                                .repeat(1, (R - n_styles_pert), 1)
                            )
                    else:
                        w_pert = new_w
                        
        
                    w_new = w_start + torch.mul(gammas.view(1, 14, 1) * w_pert, torch.sqrt(torch.tensor(dt)))
                else:
                    raise ValueError(
                        f"sampling rule unknown, should be 'stochastic' or 'extrapolation' but is {sample_rule}"
                    )

                assert torch.isfinite(w_new).all()
                if verbose:
                    print("wnew", w_new.shape)
                w = w_new

                sample, _, _ = G([w.to(device)],  input_is_latent=True)

                Ens_final[k * per_cond : (k + 1) * per_cond] = sample.detach().cpu().numpy()
                w_final[k * per_cond : (k + 1) * per_cond] = w.detach().cpu().numpy()

        return Ens_final[:N_samples], w_final
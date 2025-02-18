#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import math
from torch import optim
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm
from inversion.perceptual_loss.perceptual_loss import PerceptualLoss
from inversion.plotter import online_inv_plot, online_inv_plot, create_frame, latent_evolution_plot
from inversion.experimental_loss.ssim import ssim, ms_ssim, SSIM, MS_SSIM
from inversion.perceptual_loss.lpips.lpips import LPIPS
import utils.utils as utils
from copy import deepcopy
import time

def noise_regularize(noises):
    r'''
    Regularization of noise  
    
    Inputs : noises (list of torch.Tensor) size : TODO

    Outputs : loss (float)

    '''
    loss = 0
    for noise in noises:
        size = noise.shape[2]
        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )
            if size <= 8:
                break
            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2
    return loss


def noise_normalize_(noises):
    r''' Normalizing Noise 
    
        Input : noises (list of torch.Tensor) size : TODO
        
        Output : None

    '''
    for noise in noises:
        mean = noise.mean()
        std = noise.std()
        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    r''' Get the learning rate w.r.t the scheduling process 

        Input : 
            t (float) : step of training
            initial_lr (float) : initial learning rate
            rampdown (float) : coef. for down ramp
            rampup (float) : coef. for up ramp

    '''
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    r'''
    Adding noise to latent
    '''
    noise = torch.randn_like(latent) * strength
    return latent + noise

def feature_noise(feature, strength):
    r'''
    Adding noise to feature
    '''
    noise = torch.randn_like(feature) * strength
    return feature + noise

def optimize(Ens_r, g_ema, init_latent, device, params, Means, Maxs, Mins, features_in=None, hybrid=False, apply_log_transform=False):

    """

    Inverting Ens_r and tuning the Generator g_ema

    Inputs :

        Ens_r : torch.tensor, shape B x C x H x W

        g_ema :  stylegan Generator

        init_latent : inversion starting point
            torch.tensor, shape B x (2 log2(H) -2) x 512

            if eg H = 256, 2 log2(H) - 2 = 14
                  H = 128, 2 log2(H) - 2 = 12

    Returns :

        latent_in :  torch.tensor, same shape as latent_mean : the inverted latent codes

        noises : the vector of noises to be used


    """
    Ens_r = Ens_r.to(device) # torch.Size([B, CH, 256, 256])
    init_latent = init_latent.to(device) # torch.Size([512])
    if hybrid : 
        if len(init_latent.shape) == 2 :
            latent_in = init_latent.detach().clone().unsqueeze(0).repeat(Ens_r.shape[0], 1) # (B, 512)
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1) # (B, 14, 512)
            latent_in.requires_grad = True
        else :
            latent_in = init_latent
            latent_in.requires_grad = True
            
    else :
        latent_in = init_latent.detach().clone().unsqueeze(0).repeat(Ens_r.shape[0], 1) # (B, 512)
        latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1) # (B, 14, 512)
        latent_in.requires_grad = True
        
    if params.feature_optimize : 
        
        latent_z = torch.empty(10000, 512).normal_().to(params.device)
        with torch.no_grad():
            w = g_ema.style(latent_z)
        latent_mean = w.mean(dim=0).clone().unsqueeze(0).repeat(Ens_r.shape[0], 1).unsqueeze(1).repeat(1, g_ema.n_latent, 1)
        _, features_mean, _ = g_ema([latent_mean], input_is_latent=True, return_features=True)
        feature_mean = features_mean[params.feature_id].detach().clone().to(device)

        if features_in is None :
            _, features, _ = g_ema([latent_in], input_is_latent=True, return_features=True)
            feature = features[params.feature_id].detach().clone().to(device)
            feature.requires_grad = True
        else :
            feature = features_in.detach().clone().to(device)
            feature.requires_grad = True

    with torch.no_grad():
        noise_sample = torch.randn(Ens_r.shape[0], 512, device=device) # torch.Size([B,512]) (z)
        latent_out = g_ema.style(noise_sample) # torch.Size([B,512]) (w)
        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / Ens_r.shape[0]) ** 0.5


    print(f'########## Latent vector optimisation {params.date_index} {params.lt_index} #############')
    if params.fixed_noise or params.noise_optimize :
        noises_single = g_ema.make_noise() # list of noise maps, with shapes from (1,1,4,4) to (1,1,256,256)
        noises = [] # per pixel noise to inject in each layer. with shapes from (B,1,4,4) to (B,1,256,256)
        for i, noise in enumerate(noises_single):
            noises.append(noise.repeat(Ens_r.shape[0], 1, 1, 1).normal_())

    params_to_optimize = [latent_in]
    if params.noise_optimize:
        for noise in noises:
            noise.requires_grad = True

        params_to_optimize += noises
    
    if params.feature_optimize : 
        params_to_optimize += [feature]
    
    optimizer = optim.Adam(params_to_optimize, lr=params.lr)    

    pbar = tqdm(range(params.invstep))

    latent_path = []

    #### Perceptual Loss ####
    if params.lambda_perceptual_loss>0:
        perceptual_loss_class = PerceptualLoss(
                                        config=params,
                                        device=device,
                                        multi_scale=params.multi_scale_perceptual_loss
                                        ).to(device).eval()
        perceptual_loss_class.compute_perceptual_features(img=Ens_r)
    
    if params.lambda_lpips_loss>0:
        lpips_loss_class = LPIPS(
                                config=params,
                                device=device,
                                multi_scale=params.multi_scale_perceptual_loss
                                ).to(device).eval()
    # MS-SSIM module for MS-SSIM loss
    # ssim_module = SSIM(data_range=1, size_average=True, channel=1)
    if params.lambda_ms_ssim :
        ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3)

    list_perceptual_loss = []
    list_time_to_compute_vgg_loss=[]
    list_time_to_compute_mse_loss=[]
    features_in = None

    if params.plot_loss_evolution:
        latent_in_old = deepcopy(latent_in)
        latent_evolution=[]
        perceptual_loss_evolution=[]
        mae_loss_evolution=[]

    for i in pbar:
        loss = 0
        t = i / params.invstep
        if params.lr_rampdown ==0 and params.lr_rampup == 0:
            lr = params.lr
        else:
            lr = get_lr(t, params.lr, params.lr_rampdown, params.lr_rampup)

        optimizer.param_groups[0]["lr"] = lr

        noise_strength = latent_std * params.noise_strength * max(0, 1 - t / params.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        if params.feature_optimize : 
            features_in = [None]*(params.feature_id)+ [feature] + [None]*(13-(params.feature_id)) 

        if params.fixed_noise or params.noise_optimize :
            img_gen, features_out, _ = g_ema([latent_n], input_is_latent=True, return_features=True, noise=noises, features_in=features_in, feature_scale=params.feature_scale)
        else :
            img_gen, features_out, _ = g_ema([latent_n], input_is_latent=True, return_features=True, noise=None, features_in=features_in, feature_scale=params.feature_scale)

        batch, channel, height, width = img_gen.shape
        # print('img_gen shape :', img_gen.shape)
        if params.noise_optimize:
            noise_loss = noise_regularize(noises)
            loss+=noise_loss*params.lambda_noise
        else :
            noise_loss = 0
        
        if params.feature_optimize : 
            feature_loss = F.mse_loss(feature, features_out[params.feature_id])
            
            # feature_loss = torch.sum((feature-feature_mean).norm(2, dim=(1, 2, 3))) / feature.shape[0]
            loss += feature_loss*params.lambda_features
        else :
            feature_loss=0

        # perceptual loss
        perceptual_loss = torch.tensor(0.).to(device)
        if params.lambda_perceptual_loss>0:
            t0 = time.time()
            perceptual_loss = perceptual_loss_class(img_gen=img_gen)
            list_time_to_compute_vgg_loss.append(time.time()-t0)
            list_perceptual_loss.append(perceptual_loss.cpu().detach().numpy())
            loss += perceptual_loss*params.lambda_perceptual_loss
        else :
            list_time_to_compute_vgg_loss.append(np.NaN)
            list_perceptual_loss.append(np.NaN)
        
        # lpips loss
        lpips_loss = torch.tensor(0.).to(device)
        if params.lambda_lpips_loss>0:
            lpips_loss = lpips_loss_class(Ens_r, img_gen)
            loss += lpips_loss*params.lambda_lpips_loss

        # ms_ssim loss
        ms_ssim_loss = torch.tensor(0.).to(device)
        if params.lambda_ms_ssim>0. : 
            ms_ssim_loss = 1 - ms_ssim_module((img_gen+1)/2, (Ens_r+1)/2)
            loss += ms_ssim_loss*params.lambda_ms_ssim

        # mae/mse/amse/wamse pixel loss
        if params.pixel_loss_type=='mse' :
            t0 = time.time()
            pixel_loss = F.mse_loss(img_gen, Ens_r)
            list_time_to_compute_mse_loss.append(time.time()-t0)
            loss += params.lambda_pixel*pixel_loss

        elif params.pixel_loss_type=='mae':
            pixel_loss = F.l1_loss(img_gen, Ens_r)
            loss+=params.lambda_pixel*pixel_loss

        elif params.pixel_loss_type=='amse':
            pixel_loss = F.mse_loss(img_gen, Ens_r) + torch.max(torch.min(Ens_r,torch.tensor(20))-img_gen,torch.tensor(0)).mean()
            loss+=params.lambda_pixel*pixel_loss 

        elif params.pixel_loss_type=='wamse':
            pixel_loss = F.mse_loss(img_gen, Ens_r) + torch.max(torch.min(Ens_r,torch.tensor(20))-img_gen,torch.tensor(0)).mean()*torch.min(Ens_r+1,torch.tensor(20)).mean()
            loss+=params.lambda_pixel*pixel_loss


        else:
            raise ValueError(f"unknown pixel_loss_type: {params.pixel_loss_type}")
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if params.plot_loss_evolution:
            latent_evolution.append(F.mse_loss(latent_in.cpu().detach(), latent_in_old.cpu().detach()).numpy())
            perceptual_loss_evolution.append(perceptual_loss.cpu().detach().numpy())
            mae_loss_evolution.append(F.l1_loss(img_gen, Ens_r).item())
            latent_in_old=deepcopy(latent_in)
        
        if params.fixed_noise or params.noise_optimize :
            noise_normalize_(noises)

        display = f'lr: {lr:.4f}'
        if params.lambda_ms_ssim>0. : 
            display += f" || ms_ssim_loss: {ms_ssim_loss.item():.6f}"
        if params.lambda_perceptual_loss>0. : 
            display += f" || perceptual_loss: {perceptual_loss.item():.6f}"
        if params.lambda_lpips_loss>0. :
            display += f" || lpips_loss: {lpips_loss.item():.6f}"
        if params.feature_optimize:
            display += f" || feature_loss: {feature_loss.item():.6f}"
        if params.lambda_pixel>0:
            display += f" || pixel_loss: {pixel_loss.item():.6f}"

        display += f" || mae_loss (test only): {F.l1_loss(img_gen, Ens_r).item():.6f}" 
        pbar.set_description((display))
        
        if (i + 1) % 100 == 0 or i==params.invstep-1:
            latent_path.append(latent_in.detach().clone())
        
        denorm_img_gen = utils.denormalize(img_gen.cpu().detach(), params.normalization, Means, Mins, Maxs, apply_log_transform=apply_log_transform)
        denorm_Ens_r = utils.denormalize(Ens_r.cpu(), params.normalization, Means, Mins, Maxs, apply_log_transform=apply_log_transform)
        

        if i+1 in params.inv_checkpoints:
            print(f"--saving checkpoint {i+1}:", params.output_dir+'w_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1))
            np.save(params.output_dir+'w_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1),latent_in.cpu().detach().numpy())
            if params.fixed_noise or params.noise_optimize :
                with open(params.output_dir+'noise_{}_{}_{}.p'.format(params.date_index,params.lt_index,i+1), 'wb') as f:
                    pickle.dump({j : n.cpu().detach().numpy() for j,n in enumerate(noises)},f)
            

            if params.save_normalized_sample:
                np.save(params.output_dir+'invertFsemble_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1), img_gen.cpu().detach().numpy())
            else :
                np.save(params.output_dir+'invertFsemble_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1), denorm_img_gen)
            
            if params.feature_optimize:
                np.save(params.output_dir+'feature_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1),feature.cpu().detach().numpy())
            if params.plot_checkpoint :
                figname = params.output_dir + f"{params.date_index}_{params.lt_index}_step_{i+1}.png"
                print(f"--plotting checkpoint {i+1}: {figname}")
                figtitle = f"{params.date_index}_{params.lt_index}_step_{i+1}"
                online_inv_plot(denorm_Ens_r.cpu().detach().numpy(), denorm_img_gen.cpu().detach().numpy(), figtitle=figtitle, figname=figname)
    if params.plot_loss_evolution:
        latent_evolution_plot(latent_evolution=latent_evolution,
                            perceptual_loss_evolution=perceptual_loss_evolution,
                            mae_loss_evolution=mae_loss_evolution,
                            figtitle=f"Latent evolution during optimization for {params.date_index}_{params.lt_index}",
                            figname = params.output_dir + f"metric_evolution_{params.date_index}_{params.lt_index}.png")
        np.save(params.output_dir+'latent_evolution_{}_{}.npy'.format(params.date_index,params.lt_index),latent_evolution)
        np.save(params.output_dir+'perceptual_loss_evolution_{}_{}.npy'.format(params.date_index,params.lt_index),perceptual_loss_evolution)
    #     # gif
    #     if params.plot_gif  :
    #         if i==0 :
    #             frames = []
    #         figname = params.output_dir + f"{params.date_index}_{params.lt_index}_step_{i}.png"
    #         figtitle = f"{params.date_index}_{params.lt_index}_step_{i+1}"
    #         fig = online_inv_plot(denorm_Ens_r.cpu().detach().numpy(), denorm_img_gen.cpu().detach().numpy(), figtitle=figtitle, figname=figname, savefig=False)
    #         frames.append(create_frame(fig))

        
    # if params.plot_gif:
    #     # Just for the love of GIFs
    #     frame_one = frames[0]
    #     frame_one.save(
    #         params.output_dir + f"plot_time_step_period_{params.date_index}_{params.lt_index}.gif",
    #         format="GIF",
    #         append_images=frames,
    #         save_all=True,
    #         duration=params.invstep*20,
    #         loop=0,
    #     )
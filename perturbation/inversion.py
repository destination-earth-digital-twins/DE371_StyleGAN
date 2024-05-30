#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import math
from torch import optim
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm
from perturbation.lpips import VGGPerceptualLoss
from perturbation.plotter import online_inv_plot_2, online_inv_plot

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



def optimize(Ens_r, g_ema, latent_mean, device, params):

    """

    Inverting Ens_r and tuning the Generator g_ema

    Inputs :

        Ens_r : torch.tensor, shape B x C x H x W

        g_ema :  stylegan Generator

        latent_mean : inversion starting point
            torch.tensor, shape B x (2 log2(H) -2) x 512

            if eg H = 256, 2 log2(H) - 2 = 14
                  H = 128, 2 log2(H) - 2 = 12

    Returns :

        latent_in :  torch.tensor, same shape as latent_mean : the inverted latent codes

        noises : the vector of noises to be used


    """
    Ens_r = Ens_r.to(device) # torch.Size([B, CH, 256, 256])
    latent_mean = latent_mean.to(device) # torch.Size([512])
    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(Ens_r.shape[0], 1) # torch.Size([B, 512])

    with torch.no_grad():
        noise_sample = torch.randn(Ens_r.shape[0], 512, device=device) # torch.Size([B,512]) (z)
        latent_out = g_ema.style(noise_sample) # torch.Size([B,512]) (w)
        latent_mean = latent_out.mean(0) # mkl: this is weird. latent mean is passed as an input, but we are not using it ?
        latent_std = ((latent_out - latent_mean).pow(2).sum() / Ens_r.shape[0]) ** 0.5

    print(f'########## Latent vector optimisation {params.date_index} {params.lt_index} #############')

    noises_single = g_ema.make_noise() # list of noise maps, with shapes from (1,1,4,4) to (1,1,256,256)
    noises = [] # per pixel noise to inject in each layer. with shapes from (B,1,4,4) to (B,1,256,256)
    for i, noise in enumerate(noises_single):
        noises.append(noise.repeat(Ens_r.shape[0], 1, 1, 1).normal_())

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(Ens_r.shape[0], 1) # (B, 512)
    latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1) # (B, 14, 512)

    latent_in.requires_grad = True
    if params.noise_optimize:
        for noise in noises:
            noise.requires_grad = True

    if params.noise_optimize:
        optimizer = optim.Adam([latent_in] + noises, lr=params.lr)
    else:
        optimizer = optim.Adam([latent_in], lr=params.lr)

    pbar = tqdm(range(params.invstep))

    latent_path = []
    if params.lambda_vgg>0 :
        VGG_loss = VGGPerceptualLoss(pre_trained=params.vgg_pre_trained, init_layer=params.vgg_init_layer)
        VGG_loss.to(device)

    for i in pbar:
        t = i / params.invstep
        lr = get_lr(t, params.lr)
        optimizer.param_groups[0]["lr"] = lr

        noise_strength = latent_std * params.noise_strength * max(0, 1 - t / params.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item()) # mkl: why add noise to latent_in, seems totally unecessary??

#        if not params.noise_optimize:
#            Gen = g_ema([latent_n], input_is_latent=True, noise=None)
#        else:
#            Gen = g_ema([latent_n], input_is_latent=True, noise=noises)

        Gen = g_ema([latent_n], input_is_latent=True, noise=noises)

        img_gen = Gen[0] # generated samples

        batch, channel, height, width = img_gen.shape

        if params.noise_optimize:
            noise_loss = noise_regularize(noises)
        else:
            noise_loss = 0.

        # compute vgg/perceptual loss
        perceptual_loss = torch.tensor(0.).to(device)
        if params.lambda_vgg>0.:
            for i_mem in range(img_gen.shape[0]): # mkl: i think we can avoid double for loop by passing all members for each variable as input
                for i_var in range(img_gen.shape[1]):
                    perceptual_loss += VGG_loss( (img_gen[i_mem, i_var, :, :]+1)/2,
                                                 (Ens_r[i_mem, i_var, :, :]+1)/2,
                                                 feature_layers = params.vgg_feature_layers,
                                                 style_layers = params.vgg_style_layers,
                                                 alpha_feature = params.vgg_alpha_feature,
                                                 alpha_style = params.vgg_alpha_style )
            perceptual_loss /= img_gen.shape[0]*img_gen.shape[1]

        # compute mae/mse pixel loss
        if params.pixel_loss_type=='mse' :
            pixel_loss = F.mse_loss(img_gen, Ens_r)

        elif params.pixel_loss_type=='mae':
            pixel_loss = F.l1_loss(img_gen, Ens_r)

        else:
            raise ValueError(f"unknown pixel_loss_type: {params.pixel_loss_type}")

        # compute total loss
        loss = params.lambda_noise*noise_loss + params.lambda_pixel*pixel_loss + params.lambda_vgg*perceptual_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if params.noise_normalize:
            noise_normalize_(noises) # mkl: why do we normalize the noise?

        pbar.set_description(
            (
                f" pixel_loss: {pixel_loss.item():.6f}; lr: {lr:.4f} || perceptual_loss: {perceptual_loss.item():.6f}; lr: {lr:.4f}"
            )
        )
        if (i + 1) % 100 == 0 or i==params.invstep-1:
            latent_path.append(latent_in.detach().clone())

        if i+1 in params.inv_checkpoints:
            print(f"--saving checkpoint {i+1}:", params.output_dir+'w_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1))
            np.save(params.output_dir+'w_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1),latent_in.cpu().detach().numpy())

            with open(params.output_dir+'noise_{}_{}_{}.p'.format(params.date_index,params.lt_index,i+1), 'wb') as f:
                pickle.dump({j : n.cpu().detach().numpy() for j,n in enumerate(noises)},f)

            np.save(params.output_dir+'invertFsemble_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1),img_gen.cpu().detach().numpy())

            figname = params.output_dir + f"{params.date_index}_{params.lt_index}_step_{i+1}.png"
            print(f"--plotting checkpoint {i+1}: {figname}")
            figtitle = f"{params.date_index}_{params.lt_index}_step_{i+1}"
            online_inv_plot_2(Ens_r.cpu().detach().numpy(), img_gen.cpu().detach().numpy(), figtitle=figtitle, figname=figname)



def optimize_noise(Ens_r, g_ema, device, params, w):
    print("noise optimization, keeping w fixed")

    w = w.to(device)
    Ens_r = Ens_r.to(device)

    noises_single = g_ema.make_noise()
    noises = [] # per pixel noise to inject in each layer
    for i, noise in enumerate(noises_single):
        noises.append(noise.repeat(Ens_r.shape[0], 1, 1, 1).normal_())

    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam(noises, lr=params.lr)
    pbar = tqdm(range(params.invstep))
    VGG_loss = VGGPerceptualLoss()
    VGG_loss.to(device)

    for i in pbar:
        t = i / params.invstep
        lr = get_lr(t, params.lr)
        optimizer.param_groups[0]["lr"] = lr
        Gen = g_ema([w], input_is_latent=True, noise=noises)

        img_gen = Gen[0] # generated samples

        batch, channel, height, width = img_gen.shape

        if params.pixel_loss_type=='mse' :
            noise_loss = noise_regularize(noises)
            pixel_loss = F.mse_loss(img_gen, Ens_r)
            loss = params.lambda_noise*noise_loss + params.lambda_pixel*pixel_loss

        elif params.pixel_loss_type=='mae':
            noise_loss = noise_regularize(noises)
            pixel_loss = F.l1_loss(img_gen, Ens_r)
            loss = params.lambda_noise*noise_loss + params.lambda_pixel*pixel_loss

        perceptual_loss = 0.
        if params.lambda_vgg>0:
            for i_mem in range(img_gen.shape[0]):
                    for i_var in range(img_gen.shape[1]):
                        perceptual_loss += VGG_loss((img_gen[i_mem, i_var, :, :]+1)/2, (Ens_r[i_mem, i_var, :, :]+1)/2, feature_layers=params.vgg_feature_layers, style_layers=params.vgg_style_layers)
            perceptual_loss /= img_gen.shape[0]*img_gen.shape[1]

        loss += perceptual_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if params.noise_normalize:
            noise_normalize_(noises)

        pbar.set_description(
            (
                f" loss: {loss.item():.6f}; lr: {lr:.4f}"
            )
        )

        if i+1 in params.inv_checkpoints:
            print(f"--saving checkpoint {i+1}")

            # save noise
            with open(params.output_dir+'noise_{}_{}_{}.p'.format(params.date_index,params.lt_index,i+1), 'wb') as f:
                pickle.dump({j : n.cpu().detach().numpy() for j,n in enumerate(noises)},f)

            # save inverted samples
            np.save(params.output_dir+'invertFsemble_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1),img_gen.cpu().detach().numpy())


            # plot inverted samples
            figname = params.output_dir + f"{params.date_index}_{params.lt_index}_step_{i+1}_.png"
            print(f"--plotting checkpoint {i+1}: {figname}")
            figtitle = f"{params.date_index}_{params.lt_index}_step_{i+1}"
            online_inv_plot(Ens_r.cpu().detach().numpy(), img_gen.cpu().detach().numpy(), figtitle=figtitle, figname=figname)
    return
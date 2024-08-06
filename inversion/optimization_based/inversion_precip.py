#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:22:39 2023

@author: brochetc
"""

import torch
import math
from torch import optim
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm



def noise_regularize(noises):
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
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
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
            
    Returns :
        
        latent_in :  torch.tensor, same shape as latent_mean : the inverted latent codes
        
        noises : the vector of noises to be used
        
    
    """
    #Ens_r = Ens_r[0:2] for faster tests...
    Ens_r = Ens_r.to(device)
    
    latent_mean = latent_mean.to(device).contiguous()
    
    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(Ens_r.shape[0], 1).contiguous()
    
    with torch.no_grad():
        noise_sample = torch.randn(Ens_r.shape[0], 512, device=device).contiguous()
        latent_out = g_ema.style(noise_sample).contiguous()
    
        latent_mean = latent_out.mean(0).contiguous()
        latent_std = ((latent_out - latent_mean).pow(2).sum() / Ens_r.shape[0]) ** 0.5
    

    ###########################  FIRST STEP : latent vector optimization
    
    print(f'########## Latent vector optimisation {params.date_index} {params.lt_index} #############')
    
    noises_single = g_ema.make_noise()
    noises = []
    
    for noise in noises_single:
        noises.append(noise.repeat(Ens_r.shape[0], 1, 1, 1).normal_())
    
    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(Ens_r.shape[0], 1).contiguous()
    
    latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1).contiguous()
    
    latent_in.requires_grad = True
    
    for noise in noises:
        noise.requires_grad = True
    
    optimizer = optim.Adam([latent_in] + noises, lr=params.lr)
    
    pbar = tqdm(range(params.invstep))

    latent_path = []
    pixel_scores = []
    
    
    for i in pbar:
        print('JE SUIS ',i)
        t = i / params.invstep
        
        lr = get_lr(t, params.lr)
        optimizer.param_groups[0]["lr"] = lr
        
        noise_strength = latent_std * params.noise * max(0, 1 - t / params.noise_ramp) ** 2
        
        latent_n = latent_noise(latent_in.contiguous(), noise_strength.item())
    
        Gen = g_ema([latent_n.contiguous()], input_is_latent=True, noise = noises)

        img_gen = Gen[0].contiguous()
    
        batch, channel, height, width = img_gen.shape
    
        n_loss = noise_regularize(noises)
        with torch.no_grad():
            mse_loss = F.mse_loss(img_gen, Ens_r)
        if params.pixel_loss_type=='mse' :
        
            pixel_loss = F.mse_loss(img_gen, Ens_r)
            
        elif params.pixel_loss_type=='mae' :
            
            pixel_loss = F.l1_loss(img_gen, Ens_r)

        elif params.pixel_loss_type=='mae_std' :
            
            pixel_loss = F.l1_loss(img_gen, Ens_r) + F.l1_loss(img_gen.std(dim=0), Ens_r.std(dim=0)) 
        elif params.pixel_loss_type=='wmse':
            
            pixel_loss = (F.mse_loss(img_gen, Ens_r)*torch.min(Ens_r+1,torch.tensor(20))).mean()
        
        elif params.pixel_loss_type=='amse':
            
            pixel_loss = F.mse_loss(img_gen, Ens_r) + torch.max(torch.min(Ens_r,torch.tensor(20))-img_gen,torch.tensor(0)).mean()
            
        elif params.pixel_loss_type=='wamse':
            
            pixel_loss = F.mse_loss(img_gen, Ens_r) + torch.max(torch.min(Ens_r,torch.tensor(20))-img_gen,torch.tensor(0)).mean()*torch.min(Ens_r+1,torch.tensor(20)).mean()
            
        elif params.pixel_loss_type=='sum_pixel_loss':
            x = Ens_r.contiguous()
            y = img_gen.contiguous()
            pixel_loss = torch.abs((x-y).sum())
            
        elif params.pixel_loss_type =='sum_pixel_loss_mae':
            if i <995:
                pixel_loss = F.l1_loss(img_gen, Ens_r)
            else:
                x = Ens_r.contiguous()
                y = img_gen.contiguous()
                pixel_loss = torch.abs((x-y).sum())
                    
        elif params.pixel_loss_type =='mul_pixel_loss_mae':

            pixel_loss_mse = F.l1_loss(img_gen, Ens_r)
            x = Ens_r.contiguous()
            y = img_gen.contiguous()
            pixel_loss_sum = torch.abs((x-y).sum())
            pixel_loss = params.lambda_pixel * pixel_loss_mse + 0.00002* pixel_loss_sum
            
        elif params.pixel_loss_type =='mul_pixel_loss_mse':

            pixel_loss_mse = F.mse_loss(img_gen, Ens_r)
            x = Ens_r.contiguous()
            y = img_gen.contiguous()
            pixel_loss_sum = torch.abs((x-y).sum())
            pixel_loss = params.lambda_pixel * pixel_loss_mse + 0.00002* pixel_loss_sum
    
            

        
        loss = params.noise_regularize * n_loss\
            + params.lambda_pixel * pixel_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        noise_normalize_(noises)
        pbar.set_description(
            (
                f" pixel_loss: {mse_loss.item():.6f}; lr: {lr:.4f}"

            )
        )
        if (i + 1) % 100 == 0 or i==params.invstep-1:
            #print(i, pixel_loss.item())
            latent_path.append(latent_in.detach().clone())
            
        """if t in [0.25,0.5,0.75] :
        
            for i in range(3) :
                cmap = 'viridis' if i!=2 else 'coolwarm'
                plt.imshow(img_gen[0,i].detach().cpu().numpy(), origin = 'lower', cmap = cmap)
                plt.colorbar()
                plt.savefig(data_dir + var[i]+ '_projected_{}.png'.format(t))
                plt.close()"""
        
        pixel_scores.append(pixel_loss.item())

        data = {'params' : params, 'pixel_loss {}'.format(params.pixel_loss_type) : pixel_scores}
        
        if i+1 in params.inv_checkpoints:
            np.save(params.output_dir+'w_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1),latent_in.cpu().detach().numpy())
            
            with open(params.output_dir+'noise_{}_{}_{}.p'.format(params.date_index,params.lt_index,i+1), 'wb') as f:
                pickle.dump({j : n.cpu().detach().numpy() for j,n in enumerate(noises)},f)
            
    
            np.save(params.output_dir+'invertFsemble_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1),img_gen.cpu().detach().numpy())
            
            name = f'step_{i+1}_lr_{params.lr}_noise_{params.noise}_noisereg_{params.noise_regularize}_{params.date_index}{params.lt_index}'
            with open(params.output_dir + name + '.p', 'wb') as f :
                pickle.dump(data,f)
        
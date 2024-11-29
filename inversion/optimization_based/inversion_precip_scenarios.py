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
import os 
from inversion.lpips import VGGPerceptualLoss
#from inversion.plotter import online_inv_plot_2, online_inv_plot
import time

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

def compute_perceptual_features(img, VGG_loss, device, params):
    features = []
    styles = []
    if params.lambda_vgg>0. :
        if params.vgg_computation=='sol1':
            for i_mem in range(img.shape[0]):
                for i_var in range(img.shape[1]):
                    feature, style = VGG_loss.forward_single_img(
                                                (img[i_mem, i_var, :, :]+1)/2,
                                                feature_layers = params.vgg_feature_layers,
                                                style_layers = params.vgg_style_layers
                    )
                    features.append(feature)
                    styles.append(style)

        elif params.vgg_computation in ['sol2', 'sol4', 'sol5']:
            if params.pixel_rr_vgg_others:
            
                for i_var in range(1,4):
                    feature, style = VGG_loss.forward_single_img(
                                                        (img[:, i_var, :, :]+1)/2,
                                                        feature_layers = params.vgg_feature_layers,
                                                        style_layers = params.vgg_style_layers
                            )
                    features.append(feature)
                    styles.append(style)
            else:
                for i_var in range(img.shape[1]):
                    feature, style = VGG_loss.forward_single_img(
                                                        (img[:, i_var, :, :]+1)/2,
                                                        feature_layers = params.vgg_feature_layers,
                                                        style_layers = params.vgg_style_layers
                            )
                    features.append(feature)
                    styles.append(style)

        elif params.vgg_computation == 'sol3':
            features, styles = VGG_loss.forward_single_img(
                                                    (img+1)/2,
                                                    feature_layers = params.vgg_feature_layers,
                                                    style_layers = params.vgg_style_layers
                        )


        return features, styles
    else :
        raise NotImplementedError
    
    

def optimize(batch_dir,Ens_r,batch_idx, g_ema, latent_mean, device, params,scenario):
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

    noises_single = g_ema.make_noise() # list of noise maps, with shapes from (1,1,4,4) to (1,1,256,256)
    if params.fixed_noise or params.noise_optimize :
        noises = [] # per pixel noise to inject in each layer. with shapes from (B,1,4,4) to (B,1,256,256)
        for i, noise in enumerate(noises_single):
            noises.append(noise.repeat(Ens_r.shape[0], 1, 1, 1).normal_())

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(Ens_r.shape[0], 1) # (B, 512)
    latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1) # (B, 14, 512)
    latent_in.requires_grad = True

    if params.noise_optimize:
        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in] + noises, lr=params.lr)
    else:
        optimizer = optim.Adam([latent_in], lr=params.lr)

    pbar = tqdm(range(params.invstep))

    latent_path = []
    # if params.lambda_lpips and params.lambda_vgg:
    #     raise NotImplementedError
    
    if params.lambda_vgg>0 :
        VGG_loss = VGGPerceptualLoss(
                        state_dict_path=params.vgg_state_dict_path,
                        init_layer=True if params.vgg_computation=='sol4' else False,
                        vgg_single_channel_input=True if params.vgg_computation=='sol5' else False
        ).to(device)

        
    if params.lambda_lpips>0:
        LPIPS_loss = lpips.LPIPS(
            net=params.lpips_pnet, 
            model_path=params.lpips_linear_layers_state_dict_path, # linear layer linked to lpips
            pretrained=True, # linear layer linked to lpips
            pnet_rand_path=params.lpips_pnet_state_dict_path, # Perceptual Net Path
            pnet_tune=params.lpips_pnet_tune,
            lpips=params.lpips_mode
        ).to(device)
        if params.lpips_pnet_tune:
            optimizer.add_param_group({'params':LPIPS_loss.net.parameters()})
        # params.lpips_pnet_state_dict_path

    if params.optimize_features_computation :    
        Ens_r_features, Ens_r_styles = compute_perceptual_features(img=Ens_r, VGG_loss=VGG_loss, device=device, params=params)

    list_perceptual_loss = []
    list_pixel_loss = []
    list_time_to_compute_vgg_loss=[]
    list_time_to_compute_mse_loss=[]

    latent_path = []
    pixel_scores = []
    
    
    for i in pbar:
       # print('JE SUIS ',i)
        t = i / params.invstep
        
        lr = get_lr(t, params.lr)
        optimizer.param_groups[0]["lr"] = lr
        
        noise_strength = latent_std * params.noise * max(0, 1 - t / params.noise_ramp) ** 2
        
        latent_n = latent_noise(latent_in.contiguous(), noise_strength.item())
    
        if params.fixed_noise or params.noise_optimize :
            Gen = g_ema([latent_n], input_is_latent=True, noise=noises)
        else :
            Gen = g_ema([latent_n], input_is_latent=True, noise=None)
        #Gen = g_ema([latent_n.contiguous()], input_is_latent=True, noise = noises)

        img_gen = Gen[0].contiguous()
    
        batch, channel, height, width = img_gen.shape
        if params.noise_optimize:
            noise_loss = noise_regularize(noises)
        else :
            noise_loss = 0
        # compute vgg/perceptual loss
        perceptual_loss = torch.tensor(0.).to(device)
        if i >= params.vgg_loss_after_step and (params.lambda_vgg>0. or params.lambda_lpips>0.):
                t0 = time.time()
                if not params.optimize_features_computation : 
                    if params.vgg_computation=='sol1':
                        perceptual_loss = torch.tensor(0.).to(device)
                        for i_mem in range(img_gen.shape[0]):
                            for i_var in range(img_gen.shape[1]):
                                if params.lambda_vgg>0. :
                                    perceptual_loss += VGG_loss( (img_gen[i_mem, i_var, :, :]+1)/2,
                                                                (Ens_r[i_mem, i_var, :, :]+1)/2,
                                                                feature_layers = params.vgg_feature_layers,
                                                                style_layers = params.vgg_style_layers,
                                                                alpha_feature = params.vgg_alpha_feature,
                                                                alpha_style = params.vgg_alpha_style
                                    )
                                elif params.lambda_lpips>0. :
                                    perceptual_loss += torch.sum(torch.abs(LPIPS_loss.forward(img_gen[i_mem, i_var, :, :], Ens_r[i_mem, i_var, :, :])))
                                else:
                                    raise NotImplementedError
                        perceptual_loss /= img_gen.shape[0]*img_gen.shape[1]
                    elif params.vgg_computation in ['sol2', 'sol4', 'sol5']:
                        perceptual_loss = torch.tensor(0.).to(device)
                        if params.pixel_rr_vgg_others:
                            
                            for i_var in range(1,4):
                                if params.lambda_vgg>0. :
                                    perceptual_loss += VGG_loss( (img_gen[:, i_var, :, :]+1)/2,
                                                                    (Ens_r[:, i_var, :, :]+1)/2,
                                                                    feature_layers = params.vgg_feature_layers,
                                                                    style_layers = params.vgg_style_layers,
                                                                    alpha_feature = params.vgg_alpha_feature,
                                                                    alpha_style = params.vgg_alpha_style
                                        )
                                elif params.lambda_lpips>0. :
                                    gen = img_gen[:, i_var, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
                                    original = Ens_r[:, i_var, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
                                    perceptual_loss += torch.sum(torch.abs(LPIPS_loss.forward(gen,original)))
                                else:
                                    raise NotImplementedError
                                
                            perceptual_loss /= 3
                        else:
                            for i_var in range(img_gen.shape[1]):
                                if params.lambda_vgg>0. :
                                    perceptual_loss += VGG_loss( (img_gen[:, i_var, :, :]+1)/2,
                                                                    (Ens_r[:, i_var, :, :]+1)/2,
                                                                    feature_layers = params.vgg_feature_layers,
                                                                    style_layers = params.vgg_style_layers,
                                                                    alpha_feature = params.vgg_alpha_feature,
                                                                    alpha_style = params.vgg_alpha_style
                                        )
                                elif params.lambda_lpips>0. :
                                    gen = img_gen[:, i_var, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
                                    original = Ens_r[:, i_var, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
                                    perceptual_loss += torch.sum(torch.abs(LPIPS_loss.forward(gen,original)))
                                else:
                                    raise NotImplementedError
                                
                            perceptual_loss /= img_gen.shape[1]

                    elif params.vgg_computation == 'sol3':
                        if params.lambda_vgg>0. :
                            perceptual_loss = VGG_loss((img_gen+1)/2,
                                                        (Ens_r+1)/2,
                                                        feature_layers = params.vgg_feature_layers,
                                                        style_layers = params.vgg_style_layers,
                                                        alpha_feature = params.vgg_alpha_feature,
                                                        alpha_style = params.vgg_alpha_style
                            )
                        elif params.lambda_lpips>0. :
                            perceptual_loss =  LPIPS_loss.forward(img_gen, Ens_r)
                            perceptual_loss = torch.sum(torch.abs(perceptual_loss))
                        else:
                            raise NotImplementedError
                else :
                     
                    
                    if params.vgg_computation=='sol1':
                        for i_mem in range(img_gen.shape[0]):
                            for i_var in range(img_gen.shape[1]):
                                features_input_img=Ens_r_features[i_mem+i_var]
                                if params.vgg_style_layers:
                                    styles_input_img=Ens_r_styles[i_mem+i_var]
                                else :
                                    styles_input_img=None
                                perceptual_loss += VGG_loss.forward_given_features(
                                    target_img=(img_gen[i_mem, i_var, :, :]+1)/2,
                                    features_input_img=features_input_img, 
                                    styles_input_img=styles_input_img,
                                    feature_layers = params.vgg_feature_layers,
                                    style_layers = params.vgg_style_layers,
                                    alpha_feature = params.vgg_alpha_feature,
                                    alpha_style = params.vgg_alpha_style
                                )
                                
                        perceptual_loss /= img_gen.shape[0]*img_gen.shape[1]
                    elif params.vgg_computation in ['sol2', 'sol4', 'sol5']:
                        if params.pixel_rr_vgg_others:
                            for i_var in range(1,4):
                                features_input_img=Ens_r_features[i_var]
                                if params.vgg_style_layers:
                                    styles_input_img=Ens_r_styles[i_var]
                                else :
                                    styles_input_img=None
                                perceptual_loss += VGG_loss.forward_given_features(
                                    target_img=(img_gen[:, i_var, :, :]+1)/2,
                                    features_input_img=features_input_img, 
                                    styles_input_img=styles_input_img,
                                    feature_layers = params.vgg_feature_layers,
                                    style_layers = params.vgg_style_layers,
                                    alpha_feature = params.vgg_alpha_feature,
                                    alpha_style = params.vgg_alpha_style
                                )
                            perceptual_loss /= 3
                        else:
                            for i_var in range(img_gen.shape[1]):
                                features_input_img=Ens_r_features[i_var]
                                if params.vgg_style_layers:
                                    styles_input_img=Ens_r_styles[i_var]
                                else :
                                    styles_input_img=None
                                perceptual_loss += VGG_loss.forward_given_features(
                                    target_img=(img_gen[:, i_var, :, :]+1)/2,
                                    features_input_img=features_input_img, 
                                    styles_input_img=styles_input_img,
                                    feature_layers = params.vgg_feature_layers,
                                    style_layers = params.vgg_style_layers,
                                    alpha_feature = params.vgg_alpha_feature,
                                    alpha_style = params.vgg_alpha_style
                                )
                            perceptual_loss /= img_gen.shape[1]
                            
                    else :
                        perceptual_loss += VGG_loss.forward_given_features(
                                target_img=(img_gen+1)/2,
                                features_input_img=Ens_r_features, 
                                styles_input_img=Ens_r_styles,
                                feature_layers = params.vgg_feature_layers,
                                style_layers = params.vgg_style_layers,
                                alpha_feature = params.vgg_alpha_feature,
                                alpha_style = params.vgg_alpha_style
                        )

                list_time_to_compute_vgg_loss.append(time.time()-t0)
                list_perceptual_loss.append(perceptual_loss.cpu().detach().numpy())
        else :
            list_time_to_compute_vgg_loss.append(np.NaN)
            list_perceptual_loss.append(np.NaN)
        
        
        #n_loss = noise_regularize(noises)
        # with torch.no_grad():
        #     mse_loss = F.mse_loss(img_gen, Ens_r)
        if params.pixel_rr_vgg_others:
            if params.pixel_loss_type=='mse' :
            
                pixel_loss = F.mse_loss(img_gen[:,0,:,:], Ens_r[:,0,:,:])
                
            elif params.pixel_loss_type=='mae' :
                
                pixel_loss = F.l1_loss(img_gen[:,0,:,:], Ens_r[:,0,:,:])


            
            elif params.pixel_loss_type=='amse':
                # version avec b=20
                pixel_loss = F.mse_loss(img_gen[:,0,:,:], Ens_r[:,0,:,:]) + torch.max(torch.min(Ens_r[:,0,:,:],torch.tensor(20))-img_gen[:,0,:,:],torch.tensor(0)).mean()


            elif params.pixel_loss_type=='wamse':
                
                pixel_loss = F.mse_loss(img_gen[:,0,:,:], Ens_r[:,0,:,:]) + torch.max(torch.min(Ens_r[:,0,:,:],torch.tensor(20))-img_gen[:,0,:,:],torch.tensor(0)).mean()*torch.min(Ens_r[:,0,:,:]+1,torch.tensor(20)).mean()

                
            else:
                raise ValueError(f"unknown pixel_loss_type: {params.pixel_loss_type}")
        else:
            
            if params.pixel_loss_type=='mse' :
            
                pixel_loss = F.mse_loss(img_gen, Ens_r)
                
            elif params.pixel_loss_type=='mae' :
                
                pixel_loss = F.l1_loss(img_gen, Ens_r)

            elif params.pixel_loss_type=='mae_std' :
                
                pixel_loss = F.l1_loss(img_gen, Ens_r) + F.l1_loss(img_gen.std(dim=0), Ens_r.std(dim=0)) 
            elif params.pixel_loss_type=='wmse':
                
                pixel_loss = (F.mse_loss(img_gen, Ens_r)*torch.min(Ens_r+1,torch.tensor(20))).mean()
            
            elif params.pixel_loss_type=='amse':
                # version avec b=20
                pixel_loss = F.mse_loss(img_gen, Ens_r) + torch.max(torch.min(Ens_r,torch.tensor(20))-img_gen,torch.tensor(0)).mean()
                #version avec b = target 
                # pixel_loss = F.mse_loss(img_gen, Ens_r) + torch.max(Ens_r-img_gen,torch.tensor(0)).mean()
                # Avec b=15
                # pixel_loss = F.mse_loss(img_gen, Ens_r) + torch.max(torch.min(Ens_r,torch.tensor(15))-img_gen,torch.tensor(0)).mean()
                # b = 40
                # pixel_loss = F.mse_loss(img_gen, Ens_r) + torch.max(torch.min(Ens_r,torch.tensor(40))-img_gen,torch.tensor(0)).mean()


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
                
            else:
                raise ValueError(f"unknown pixel_loss_type: {params.pixel_loss_type}")
    
            

        
        # loss = params.noise_regularize * n_loss\
        #     + params.lambda_pixel * pixel_loss
        
        if params.lambda_vgg>0. :
            weighted_perceptual_loss = params.lambda_vgg*perceptual_loss
        
        elif params.lambda_lpips>0. :
            weighted_perceptual_loss = params.lambda_lpips*perceptual_loss
        
        else :
            weighted_perceptual_loss=0
            
        # compute total loss
        if not params.progressive_loss_mode :
            loss = params.noise_optimize*params.lambda_noise*noise_loss + params.lambda_pixel*pixel_loss + weighted_perceptual_loss
        else :
            # Todo : See if it is relevant to include the noise loss in the (1-t) part
            loss = params.noise_optimize*params.lambda_noise*noise_loss + params.lambda_pixel*pixel_loss*(1-t) + (weighted_perceptual_loss)*t


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if params.fixed_noise or params.noise_optimize :
            noise_normalize_(noises)
      #  noise_normalize_(noises)
        pbar.set_description(
            (
                f" pixel_loss: {pixel_loss.item():.6f}; lr: {lr:.4f} || perceptual_loss: {perceptual_loss.item():.6f}; lr: {lr:.4f}"
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
            np.save(os.path.join(params.output_dir,batch_dir)+'/w_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1),latent_in.cpu().detach().numpy())
            
            # with open(os.path.join(params.output_dir,batch_dir)+'/noise_{}_{}_{}.p'.format(params.date_index,params.lt_index,i+1), 'wb') as f:
            #     pickle.dump({j : n.cpu().detach().numpy() for j,n in enumerate(noises)},f)
            
    
            np.save(os.path.join(params.output_dir,scenario)+'/invertFsemble_{}_{}_{}_{}_.npy'.format(params.date_index,params.lt_index,batch_dir,i+1),img_gen.cpu().detach().numpy())
            name = f'step_{i+1}_lr_{params.lr}_noise_{params.noise}_noisereg_{params.noise_regularize}_{params.date_index}{params.lt_index}'
            with open(os.path.join(params.output_dir,batch_dir) +'/'+ name + '.p', 'wb') as f :
                pickle.dump(data,f)
        
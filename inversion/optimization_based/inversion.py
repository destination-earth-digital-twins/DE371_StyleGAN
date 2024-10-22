#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import math
from torch import optim
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm
from inversion.vgg_perceptual_loss import VGGPerceptualLoss
from inversion.patch_vgg_perceptual_loss import PatchVGGPerceptualLoss
from inversion.plotter import online_inv_plot_2, online_inv_plot
# import inversion.PerceptualSimilarity.lpips as lpips
from inversion.ssim import ssim, ms_ssim, SSIM, MS_SSIM
# from inversion.hd_vgg_perceptual_loss import VGG16ConvLoss
import time
from torch.autograd import Variable

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
        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / Ens_r.shape[0]) ** 0.5

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

    #### Perceptual Loss ####
    if params.lambda_lpips and params.lambda_vgg:
        raise NotImplementedError
    
    if params.hd_vgg :
        # VGG_loss = VGG16ConvLoss().to(device).requires_grad_(False)
        raise NotImplementedError
    else :
        if params.lambda_vgg>0 and not params.patch_mode:
            VGG_loss = VGGPerceptualLoss(
                            state_dict_path=params.vgg_state_dict_path,
                            init_layer=True if params.vgg_computation=='sol4' else False,
                            vgg_single_channel_input=True if params.vgg_computation=='sol5' else False
            ).to(device)

        elif params.lambda_vgg>0 and params.patch_mode:
            VGG_loss = PatchVGGPerceptualLoss(
                            state_dict_path=params.vgg_state_dict_path,
                            init_layer=True if params.vgg_computation=='sol4' else False,
                            vgg_single_channel_input=True if params.vgg_computation=='sol5' else False,
                            split_factor=params.split_factor
            ).to(device)

    # if params.lambda_lpips>0:
    #     LPIPS_loss = lpips.LPIPS(
    #         net=params.lpips_pnet, 
    #         model_path=params.lpips_linear_layers_state_dict_path, # linear layer linked to lpips
    #         pretrained=True, # linear layer linked to lpips
    #         pnet_rand_path=params.lpips_pnet_state_dict_path, # Perceptual Net Path
    #         pnet_tune=params.lpips_pnet_tune,
    #         lpips=params.lpips_mode
    #     ).to(device)
    #     if params.lpips_pnet_tune:
    #         optimizer.add_param_group({'params':LPIPS_loss.net.parameters()})
    #     # params.lpips_pnet_state_dict_path
    if params.hd_vgg :
        Ens_r_features=[]
        if params.vgg_computation=='sol1':
            for i_mem in range(Ens_r.shape[0]):
                for i_var in range(Ens_r.shape[1]):
                    Ens_r_features.append(VGG_loss(Ens_r[i_mem, i_var, :, :]))
        elif params.vgg_computation in ['sol2', 'sol4', 'sol5']:
            for i_var in range(Ens_r.shape[1]):
                print(np.shape(Ens_r[:, i_var, :, :]))
                Ens_r_features.append(VGG_loss(Ens_r[:, i_var, :, :]))
        elif params.vgg_computation == 'sol3':
            Ens_r_features = VGG_loss(Ens_r)
        else :
            raise NotImplementedError
        

    else :
        if params.optimize_features_computation and (params.lambda_lpips or params.lambda_vgg) :    
            Ens_r_features, Ens_r_styles = compute_perceptual_features(img=Ens_r, VGG_loss=VGG_loss, device=device, params=params)
        
    # MS-SSIM module for MS-SSIM loss
    # ssim_module = SSIM(data_range=1, size_average=True, channel=1)
    if params.lambda_ms_ssim :
        if params.vgg_computation == 'sol3':
            ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3)
        else :
            ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=1)

    list_perceptual_loss = []
    list_pixel_loss = []
    list_time_to_compute_vgg_loss=[]
    list_time_to_compute_mse_loss=[]

    for i in pbar:
        t = i / params.invstep
        lr = get_lr(t, params.lr, params.lr_rampdown, params.lr_rampup)
        optimizer.param_groups[0]["lr"] = lr

        noise_strength = latent_std * params.noise_strength * max(0, 1 - t / params.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        if params.fixed_noise or params.noise_optimize :
            Gen = g_ema([latent_n], input_is_latent=True, noise=noises)
        else :
            Gen = g_ema([latent_n], input_is_latent=True, noise=None)

        img_gen = Gen[0] # generated samples

        batch, channel, height, width = img_gen.shape
        # print('img_gen shape :', img_gen.shape)
        if params.noise_optimize:
            noise_loss = noise_regularize(noises)
        else :
            noise_loss = 0
        
        # compute vgg/perceptual loss
        perceptual_loss = torch.tensor(0.).to(device)
        ms_ssim_loss = torch.tensor(0.).to(device)
        if (i >= params.vgg_loss_after_step and (params.lambda_vgg>0. or params.lambda_lpips>0.)) or params.lambda_ms_ssim>0:
                t0 = time.time()
                if not params.optimize_features_computation : 
                    if params.hd_vgg :
                        raise NotImplementedError
                    if params.vgg_computation=='sol1':
                        for i_mem in range(img_gen.shape[0]):
                            for i_var in range(img_gen.shape[1]):
                                # Perceptual Loss
                                if params.lambda_vgg>0. :
                                    perceptual_loss += VGG_loss( (img_gen[i_mem, i_var, :, :]+1)/2,
                                                                (Ens_r[i_mem, i_var, :, :]+1)/2,
                                                                feature_layers = params.vgg_feature_layers,
                                                                style_layers = params.vgg_style_layers,
                                                                alpha_feature = params.vgg_alpha_feature,
                                                                alpha_style = params.vgg_alpha_style
                                    )
                                # elif params.lambda_lpips>0. :
                                #     perceptual_loss += torch.sum(torch.abs(LPIPS_loss.forward(img_gen[i_mem, i_var, :, :], Ens_r[i_mem, i_var, :, :])))
                                else:
                                    raise NotImplementedError
                                # MS_SSIM Loss
                                if params.lambda_ms_ssim>0. : 
                                    ms_ssim_loss += 1 - ms_ssim_module((img_gen[i_mem, i_var, :, :]+1)/2, (Ens_r[i_mem, i_var, :, :]+1)/2)

                        ms_ssim_loss /= img_gen.shape[0]*img_gen.shape[1]
                        perceptual_loss /= img_gen.shape[0]*img_gen.shape[1]
                    elif params.vgg_computation in ['sol2', 'sol4', 'sol5']:
                        perceptual_loss = torch.tensor(0.).to(device)
                        for i_var in range(img_gen.shape[1]):
                            # Perceptual Loss
                            if params.lambda_vgg>0. :
                                perceptual_loss += VGG_loss( (img_gen[:, i_var, :, :]+1)/2,
                                                                (Ens_r[:, i_var, :, :]+1)/2,
                                                                feature_layers = params.vgg_feature_layers,
                                                                style_layers = params.vgg_style_layers,
                                                                alpha_feature = params.vgg_alpha_feature,
                                                                alpha_style = params.vgg_alpha_style
                                    )
                            # elif params.lambda_lpips>0. :
                            #     gen = img_gen[:, i_var, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
                            #     original = Ens_r[:, i_var, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
                            #     perceptual_loss += torch.sum(torch.abs(LPIPS_loss.forward(gen,original)))
                            else:
                                raise NotImplementedError
                            # MS_SSIM Loss
                            if params.lambda_ms_ssim>0. : 
                                ms_ssim_loss += 1 - ms_ssim_module((img_gen[:, i_var, :, :]+1)/2, (Ens_r[:, i_var, :, :]+1)/2)

                        ms_ssim_loss /= img_gen.shape[1]
                        perceptual_loss /= img_gen.shape[1]

                    elif params.vgg_computation == 'sol3':
                        # Perceptual Loss
                        if params.lambda_vgg>0. :
                            perceptual_loss = VGG_loss((img_gen+1)/2,
                                                        (Ens_r+1)/2,
                                                        feature_layers = params.vgg_feature_layers,
                                                        style_layers = params.vgg_style_layers,
                                                        alpha_feature = params.vgg_alpha_feature,
                                                        alpha_style = params.vgg_alpha_style
                            )
                        # elif params.lambda_lpips>0. :
                        #     perceptual_loss =  LPIPS_loss.forward(img_gen, Ens_r)
                        #     perceptual_loss = torch.sum(torch.abs(perceptual_loss))
                        else:
                            raise NotImplementedError
                        
                        # MS_SSIM Loss
                        if params.lambda_ms_ssim>0. : 
                            ms_ssim_loss = 1 - ms_ssim_module((img_gen+1)/2, (Ens_r+1)/2)
                    
                    else :
                        raise NotImplementedError

                else :
                    if params.vgg_computation=='sol1':
                        for i_mem in range(img_gen.shape[0]):
                            for i_var in range(img_gen.shape[1]):
                                # Perceptual Loss
                                if params.hd_vgg :
                                    img_gen_features = VGG_loss(img_gen[i_mem, i_var, :, :])
                                    perceptual_loss += F.mse_loss(Ens_r_features[i_mem+i_var], img_gen_features)
                                else :
                                    if params.lambda_vgg>0. :
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
                                # MS_SSIM Loss
                                if params.lambda_ms_ssim>0. : 
                                    ms_ssim_loss += 1 - ms_ssim_module((img_gen[i_mem, i_var, :, :]+1)/2, (Ens_r[i_mem, i_var, :, :]+1)/2)

                        ms_ssim_loss /= img_gen.shape[0]*img_gen.shape[1]
                        perceptual_loss /= img_gen.shape[0]*img_gen.shape[1]

                    elif params.vgg_computation in ['sol2', 'sol4', 'sol5']:
                        for i_var in range(img_gen.shape[1]):
                            # Perceptual Loss
                            if params.hd_vgg :
                                img_gen_features = VGG_loss(img_gen[:, i_var, :, :])
                                perceptual_loss += F.mse_loss(Ens_r_features[i_var], img_gen_features)
                                
                            else :
                                if params.lambda_vgg>0. :
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
                            # MS_SSIM Loss
                            if params.lambda_ms_ssim>0. : 
                                # ssim_loss = 1 - ssim_module((img_gen[:, i_var, :, :]+1)/2, (Ens_r[:, i_var, :, :]+1)/2)
                                ms_ssim_loss += 1 - ms_ssim_module((img_gen[:, i_var, :, :]+1)/2, (Ens_r[:, i_var, :, :]+1)/2)

                        ms_ssim_loss /= img_gen.shape[1]
                        perceptual_loss /= img_gen.shape[1]

                    elif params.vgg_computation == 'sol3':
                        # Perceptual Loss
                        if params.hd_vgg :
                            img_gen_features = VGG_loss(img_gen)
                            perceptual_loss = F.mse_loss(Ens_r_features, img_gen_features)
                        else :
                            if params.lambda_vgg>0. :
                                perceptual_loss += VGG_loss.forward_given_features(
                                    target_img=(img_gen+1)/2,
                                    features_input_img=Ens_r_features, 
                                    styles_input_img=Ens_r_styles,
                                    feature_layers = params.vgg_feature_layers,
                                    style_layers = params.vgg_style_layers,
                                    alpha_feature = params.vgg_alpha_feature,
                                    alpha_style = params.vgg_alpha_style
                                )
                        # MS_SSIM Loss
                        if params.lambda_ms_ssim>0. : 
                            ms_ssim_loss = 1 - ms_ssim_module((img_gen+1)/2, (Ens_r+1)/2)
                    else :
                        raise NotImplementedError

                list_time_to_compute_vgg_loss.append(time.time()-t0)
                list_perceptual_loss.append(perceptual_loss.cpu().detach().numpy())
        else :
            list_time_to_compute_vgg_loss.append(np.NaN)
            list_perceptual_loss.append(np.NaN)
        

        

        # compute mae/mse pixel loss
        if params.pixel_loss_type=='mse' :
            t0 = time.time()
            pixel_loss = F.mse_loss(img_gen, Ens_r)
            list_time_to_compute_mse_loss.append(time.time()-t0)
            list_pixel_loss.append(pixel_loss.cpu().detach().numpy())

        elif params.pixel_loss_type=='mae':
            pixel_loss = F.l1_loss(img_gen, Ens_r)

        else:
            raise ValueError(f"unknown pixel_loss_type: {params.pixel_loss_type}")

        if params.lambda_vgg>0. :
            weighted_perceptual_loss = params.lambda_vgg*perceptual_loss
        # elif params.lambda_lpips>0. :
        #     weighted_perceptual_loss = params.lambda_lpips*perceptual_loss
        else :
            weighted_perceptual_loss=0
        # compute total loss
        if not params.progressive_loss_mode :
            loss = params.noise_optimize*params.lambda_noise*noise_loss + params.lambda_pixel*pixel_loss + params.lambda_ms_ssim*ms_ssim_loss + weighted_perceptual_loss
        else :
            loss = params.noise_optimize*params.lambda_noise*noise_loss + params.lambda_pixel*pixel_loss*(1-t) + (weighted_perceptual_loss+params.lambda_ms_ssim*ms_ssim_loss)*t

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if params.fixed_noise or params.noise_optimize :
            noise_normalize_(noises)

        display = f'lr: {lr:.4f}'
        if params.lambda_ms_ssim>0. : 
            display += f" || ms_ssim_loss: {ms_ssim_loss.item():.6f}"
        if params.lambda_vgg>0. or params.lambda_lpips>0. : 
            display += f" || perceptual_loss: {perceptual_loss.item():.6f}"
        if weighted_perceptual_loss: 
            display += f" || weighted_perceptual_loss: {weighted_perceptual_loss:.6f}"
        if params.lambda_pixel>0. : 
            display += f" || pixel_loss: {pixel_loss.item():.6f}"
            display += f" || weighted_pixel_loss: {params.lambda_pixel*pixel_loss.item():.6f}"
            
        if params.lambda_noise>0. : 
            display += f" || noise_loss: {noise_loss:.6f}"
            display += f" || weighted_noise_loss: {params.noise_optimize*params.lambda_noise*noise_loss:.6f}"
            
            
        pbar.set_description((display))

        if (i + 1) % 100 == 0 or i==params.invstep-1:
            latent_path.append(latent_in.detach().clone())

        if i+1 in params.inv_checkpoints:
            print(f"--saving checkpoint {i+1}:", params.output_dir+'w_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1))
            np.save(params.output_dir+'w_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1),latent_in.cpu().detach().numpy())
            if params.fixed_noise or params.noise_optimize :
                with open(params.output_dir+'noise_{}_{}_{}.p'.format(params.date_index,params.lt_index,i+1), 'wb') as f:
                    pickle.dump({j : n.cpu().detach().numpy() for j,n in enumerate(noises)},f)

            np.save(params.output_dir+'invertFsemble_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1),img_gen.cpu().detach().numpy())

            if params.plot_checkpoint :
                figname = params.output_dir + f"{params.date_index}_{params.lt_index}_step_{i+1}.png"
                print(f"--plotting checkpoint {i+1}: {figname}")
                figtitle = f"{params.date_index}_{params.lt_index}_step_{i+1}"
                online_inv_plot_2(Ens_r.cpu().detach().numpy(), img_gen.cpu().detach().numpy(), figtitle=figtitle, figname=figname)
                
            # print(f"--saving loss_function {i+1}: {figname}")
            # np.save(params.output_dir+'MSE_loss_{}_{}.npy'.format(params.date_index,params.lt_index),list_pixel_loss)
            # np.save(params.output_dir+'Perceptual_loss_{}_{}.npy'.format(params.date_index,params.lt_index),list_perceptual_loss)
            # np.save(params.output_dir+'Time_Perceptual_loss_{}_{}.npy'.format(params.date_index,params.lt_index),list_time_to_compute_vgg_loss)
            # np.save(params.output_dir+'Time_MSE_loss_{}_{}.npy'.format(params.date_index,params.lt_index),list_time_to_compute_mse_loss)


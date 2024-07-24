#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import math
from torch import optim
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm
from inversion.lpips import VGGPerceptualLoss
from inversion.plotter import online_inv_plot_2, online_inv_plot
import time

torch.autograd.set_detect_anomaly(True)

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



def optimize(Ens_r, g_ema, latent_mean, device, params,j):
    nom = j 
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
    print('JE SUIS ENSR', Ens_r.shape, type(Ens_r))
    latent_mean = latent_mean.to(device) # torch.Size([512])
    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(Ens_r.shape[0], 1) # torch.Size([B, 512])

    with torch.no_grad():
        noise_sample = torch.randn(Ens_r.shape[0], 512, device=device) # torch.Size([B,512]) (z)
        latent_out = g_ema.style(noise_sample) # torch.Size([B,512]) (w)
        latent_mean = latent_out.mean(0) # mkl: this is weird. latent mean is passed as an input, but we are not using it ?
        latent_std = ((latent_out - latent_mean).pow(2).sum() / Ens_r.shape[0]) ** 0.5

    # print(f'########## Latent vector optimisation {params.date_index} {params.lt_index} #############')

    noises_single = g_ema.make_noise() # list of noise maps, with shapes from (1,1,4,4) to (1,1,256,256)
    if params.fixed_noise or params.noise_optimize :
        noises = [] # per pixel noise to inject in each layer. with shapes from (B,1,4,4) to (B,1,256,256)
        for i, noise in enumerate(noises_single):
            noises.append(noise.repeat(Ens_r.shape[0], 1, 1, 1).normal_())

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(Ens_r.shape[0], 1) # (B, 512)
    latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1) # (B, 14, 512)
    latent_in.requires_grad = True
    Gen = g_ema([latent_in], input_is_latent=True, noise=None)
    if params.noise_optimize:
        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in] + noises, lr=params.lr)
    else:
        optimizer = optim.Adam([latent_in], lr=params.lr)

    pbar = tqdm(range(params.invstep))

    latent_path = []
    if params.lambda_vgg>0 :
        VGG_loss = VGGPerceptualLoss(
                        state_dict_path=params.vgg_state_dict_path,
                        init_layer=True if params.vgg_computation=='sol4' else False,
                        vgg_single_channel_input=True if params.vgg_computation=='sol5' else False
        )
        VGG_loss.to(device)

    list_perceptual_loss = []
    list_pixel_loss = []
    list_time_to_compute_vgg_loss=[]
    list_time_to_compute_mse_loss=[]

    for i in pbar:
        t = i / params.invstep
        lr = get_lr(t, params.lr)
        optimizer.param_groups[0]["lr"] = lr

        noise_strength = latent_std * params.noise_strength * max(0, 1 - t / params.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item()) # mkl: why add noise to latent_in, seems totally unecessary??

        if params.fixed_noise or params.noise_optimize :
            Gen = g_ema([latent_n], input_is_latent=True, noise=noises)
        else :
            Gen = g_ema([latent_n], input_is_latent=True, noise=None)

        img_gen = Gen[0] # generated samples
    
            
        
        print('ICI CESTLA DIMENSION',img_gen.shape)

        #print('ACTUELLEMENT LA',img_gen.shape,img_gen[0].shape,img_gen[0][0].shape,torch.min(img_gen[0][0]),10**torch.max(img_gen[0][0])-1)
        # normalise samples and save in pack dir. obs! make sure normalization is done correctly (according to how model was trained)
    
        batch, channel, height, width = img_gen.shape

        #print('ICI DEUXIEME DIM IMGEN',img_gen.shape,'ENSRR',Ens_r.shape,Ens_r[0],Gen,'VOILAAA',len(Gen),img_gen)
        if params.noise_optimize:
            noise_loss = noise_regularize(noises)
        else :
            noise_loss = 0

        # compute vgg/perceptual loss
        # perceptual_loss = torch.tensor(0.).to(device)
        # if i >= params.vgg_loss_after_step and params.lambda_vgg>0.:
        #         t0 = time.time()
        #         if params.vgg_computation=='sol1':
        #             perceptual_loss = torch.tensor(0.).to(device)
        #             for i_mem in range(img_gen.shape[0]): # mkl: i think we can avoid double for loop by passing all members for each variable as input
        #                 for i_var in range(img_gen.shape[1]):
        #                     perceptual_loss += VGG_loss( (img_gen[i_mem, i_var, :, :]+1)/2,
        #                                                 (Ens_r[i_mem, i_var, :, :]+1)/2,
        #                                                 feature_layers = params.vgg_feature_layers,
        #                                                 style_layers = params.vgg_style_layers,
        #                                                 alpha_feature = params.vgg_alpha_feature,
        #                                                 alpha_style = params.vgg_alpha_style
        #                     )
        #             perceptual_loss /= img_gen.shape[0]*img_gen.shape[1]
        #         elif params.vgg_computation in ['sol2', 'sol4', 'sol5']:
        #             perceptual_loss = torch.tensor(0.).to(device)
        #             for i_var in range(img_gen.shape[1]):
        #                 perceptual_loss += VGG_loss( (img_gen[:, i_var, :, :]+1)/2,
        #                                                 (Ens_r[:, i_var, :, :]+1)/2,
        #                                                 feature_layers = params.vgg_feature_layers,
        #                                                 style_layers = params.vgg_style_layers,
        #                                                 alpha_feature = params.vgg_alpha_feature,
        #                                                 alpha_style = params.vgg_alpha_style
        #                     )
        #             perceptual_loss /= img_gen.shape[1]
        #         elif params.vgg_computation == 'sol3':
        #             perceptual_loss = VGG_loss((img_gen+1)/2,
        #                                         (Ens_r+1)/2,
        #                                         feature_layers = params.vgg_feature_layers,
        #                                         style_layers = params.vgg_style_layers,
        #                                         alpha_feature = params.vgg_alpha_feature,
        #                                         alpha_style = params.vgg_alpha_style
        #             )
        #         list_time_to_compute_vgg_loss.append(time.time()-t0)
        #         list_perceptual_loss.append(perceptual_loss.cpu().detach().numpy())
        # else :
        #     list_time_to_compute_vgg_loss.append(np.NaN)
        #     list_perceptual_loss.append(np.NaN)

        # compute mae/mse pixel loss
        if params.pixel_loss_type=='mse' :
            t0 = time.time()
            pixel_loss = F.mse_loss(img_gen, Ens_r)
            print(pixel_loss, 'EXEMPLE')
            list_time_to_compute_mse_loss.append(time.time()-t0)
            list_pixel_loss.append(pixel_loss.cpu().detach().numpy())
            print('MSE',pixel_loss, 'EXEMPLE mse', type(pixel_loss))

        elif params.pixel_loss_type=='mae':
            pixel_loss = F.l1_loss(img_gen, Ens_r)
        
        elif params.pixel_loss_type=='wmse':
            t0 = time.time()
            pixel_loss = (F.mse_loss(img_gen, Ens_r)*torch.min(Ens_r+1,torch.tensor(20))).mean()
            list_time_to_compute_mse_loss.append(time.time()-t0)
            list_pixel_loss.append(pixel_loss.cpu().detach().numpy())
            print('WMSE',pixel_loss, 'EXEMPLE wmse', type(pixel_loss))

        elif params.pixel_loss_type=='amse':
            t0 = time.time()
            pixel_loss = F.mse_loss(img_gen, Ens_r) + torch.max(torch.min(Ens_r,torch.tensor(20))-img_gen,torch.tensor(0)).mean()
            list_time_to_compute_mse_loss.append(time.time()-t0)
            list_pixel_loss.append(pixel_loss.cpu().detach().numpy())
        
        elif params.pixel_loss_type=='wamse':
            t0 = time.time()
            pixel_loss = F.mse_loss(img_gen, Ens_r) + torch.max(torch.min(Ens_r,torch.tensor(20))-img_gen,torch.tensor(0)).mean()*torch.min(Ens_r+1,torch.tensor(20)).mean()
            list_time_to_compute_mse_loss.append(time.time()-t0)
            list_pixel_loss.append(pixel_loss.cpu().detach().numpy())
            
        elif params.pixel_loss_type=='sum_pixel_loss':
            t0 = time.time()
            pixel_loss = torch.sum(Ens_r.contiguous())-torch.sum(img_gen.contiguous())
            print('JE SUIS OCNTINUF',torch.sum(Ens_r.contiguous()),torch.sum(img_gen.contiguous()),F.mse_loss(img_gen, Ens_r),torch.max(torch.min(Ens_r,torch.tensor(20))))

            list_time_to_compute_mse_loss.append(time.time()-t0)
            list_pixel_loss.append(pixel_loss.cpu().detach().numpy())



        else:
            raise ValueError(f"unknown pixel_loss_type: {params.pixel_loss_type}")

        # compute total loss
        if not params.progressive_loss_mode :
            loss = params.noise_optimize*params.lambda_noise*noise_loss + params.lambda_pixel*pixel_loss #+ params.lambda_vgg*perceptual_loss
            loss =loss.contiguous()

        else :
            # Todo : See if it is relevant to include the noise loss in the (1-t) part
            loss = params.noise_optimize*params.lambda_noise*noise_loss + params.lambda_pixel*pixel_loss*(1-t) #+ params.lambda_vgg*perceptual_loss*t
            loss =loss.contiguous()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if params.fixed_noise or params.noise_optimize :
            noise_normalize_(noises)

        # pbar.set_description(
        #     (
        #         f" pixel_loss: {pixel_loss.item():.6f}; lr: {lr:.4f} || perceptual_loss: {perceptual_loss.item():.6f}; lr: {lr:.4f}"
        #     )
        # )
        if (i + 1) % 100 == 0 or i==params.invstep-1:
            latent_path.append(latent_in.detach().clone())

        if i+1 in params.inv_checkpoints:
            # print(f"--saving checkpoint {i+1}:", params.output_dir+'w_{j}_{}_{}.npy')#.format(params.date_index,params.lt_index,i+1))
            # np.save(params.output_dir+f'w_{j}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1),latent_in.cpu().detach().numpy())#.format(params.date_index,params.lt_index,i+1),latent_in.cpu().detach().numpy())
            # if params.fixed_noise or params.noise_optimize :
            #     with open(params.output_dir+f'noise_{j}_{}_{}.p','wb') as f: #.format(params.date_index,params.lt_index,i+1), 'wb') as f:
            #         pickle.dump({z : n.cpu().detach().numpy() for z,n in enumerate(noises)},f)
            np.save(params.output_dir+f'invertFsemble_{j}_.npy',img_gen.cpu().detach().numpy())#.format(params.date_index,params.lt_index,i+1),img_gen.cpu().detach().numpy())

        #     figname = params.output_dir + f"{params.date_index}_{params.lt_index}_step_{i+1}.png"
        #     print(f"--plotting checkpoint {i+1}: {figname}")
        #     figtitle = f"{params.date_index}_{params.lt_index}_step_{i+1}"
        #    # print('ICI CEST MOIDDDDDDDDDDDDDDDDDDD',type(img_gen),type(Ens_r),img_gen.shape,Ens_r.shape)

        #     #online_inv_plot_2(Ens_r.cpu().detach().numpy(), img_gen.cpu().detach().numpy(), figtitle=figtitle, figname=figname)
            
        #     print(f"--saving loss_function {i+1}: {figname}")
        #     np.save(params.output_dir+'MSE_loss_{}_{}.npy'.format(params.date_index,params.lt_index),list_pixel_loss)
        #  #   np.save(params.output_dir+'Perceptual_loss_{}_{}.npy'.format(params.date_index,params.lt_index),list_perceptual_loss)
        #    # np.save(params.output_dir+'Time_Perceptual_loss_{}_{}.npy'.format(params.date_index,params.lt_index),list_time_to_compute_vgg_loss)
        #     np.save(params.output_dir+'Time_MSE_loss_{}_{}.npy'.format(params.date_index,params.lt_index),list_time_to_compute_mse_loss)
            # print(f"--saving loss_function {i+1}: {figname}")
            # np.save(params.output_dir+'MSE_loss_{}_{}.npy'.format(params.date_index,params.lt_index),list_pixel_loss)
            # np.save(params.output_dir+'Perceptual_loss_{}_{}.npy'.format(params.date_index,params.lt_index),list_perceptual_loss)
            # np.save(params.output_dir+'Time_Perceptual_loss_{}_{}.npy'.format(params.date_index,params.lt_index),list_time_to_compute_vgg_loss)
            # np.save(params.output_dir+'Time_MSE_loss_{}_{}.npy'.format(params.date_index,params.lt_index),list_time_to_compute_mse_loss)


# def optimize_noise(Ens_r, g_ema, device, params, w):
#     print("noise optimization, keeping w fixed")

#     w = w.to(device)
#     Ens_r = Ens_r.to(device)

#     noises_single = g_ema.make_noise()
#     noises = [] # per pixel noise to inject in each layer
#     for i, noise in enumerate(noises_single):
#         noises.append(noise.repeat(Ens_r.shape[0], 1, 1, 1).normal_())

#     for noise in noises:
#         noise.requires_grad = True

#     optimizer = optim.Adam(noises, lr=params.lr)
#     pbar = tqdm(range(params.invstep))
#     VGG_loss = VGGPerceptualLoss()
#     VGG_loss.to(device)

#     for i in pbar:
#         t = i / params.invstep
#         lr = get_lr(t, params.lr)
#         optimizer.param_groups[0]["lr"] = lr
#         Gen = g_ema([w], input_is_latent=True, noise=noises)

#         img_gen = Gen[0] # generated samples

#         batch, channel, height, width = img_gen.shape

#         if params.pixel_loss_type=='mse' :
#             noise_loss = noise_regularize(noises)
#             pixel_loss = F.mse_loss(img_gen, Ens_r)
#             loss = params.lambda_noise*noise_loss + params.lambda_pixel*pixel_loss

#         elif params.pixel_loss_type=='mae':
#             noise_loss = noise_regularize(noises)
#             pixel_loss = F.l1_loss(img_gen, Ens_r)
#             loss = params.lambda_noise*noise_loss + params.lambda_pixel*pixel_loss

#         perceptual_loss = 0.
#         if params.lambda_vgg>0:
#             for i_mem in range(img_gen.shape[0]):
#                     for i_var in range(img_gen.shape[1]):
#                         perceptual_loss += VGG_loss((img_gen[i_mem, i_var, :, :]+1)/2, (Ens_r[i_mem, i_var, :, :]+1)/2, feature_layers=params.vgg_feature_layers, style_layers=params.vgg_style_layers)
#             perceptual_loss /= img_gen.shape[0]*img_gen.shape[1]

#         loss += perceptual_loss

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if params.noise_normalize:
#             noise_normalize_(noises)

#         pbar.set_description(
#             (
#                 f" loss: {loss.item():.6f}; lr: {lr:.4f}"
#             )
#         )

#         if i+1 in params.inv_checkpoints:
#             print(f"--saving checkpoint {i+1}")

#             # save noise
#             with open(params.output_dir+'noise_{}_{}_{}.p'.format(params.date_index,params.lt_index,i+1), 'wb') as f:
#                 pickle.dump({j : n.cpu().detach().numpy() for j,n in enumerate(noises)},f)

#             # save inverted samples
#             np.save(params.output_dir+'invertFsemble_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i+1),img_gen.cpu().detach().numpy())


#             # plot inverted samples
#             figname = params.output_dir + f"{params.date_index}_{params.lt_index}_step_{i+1}_.png"
#             print(f"--plotting checkpoint {i+1}: {figname}")
#             figtitle = f"{params.date_index}_{params.lt_index}_step_{i+1}"
#             online_inv_plot(Ens_r.cpu().detach().numpy(), img_gen.cpu().detach().numpy(), figtitle=figtitle, figname=figname)
#     return
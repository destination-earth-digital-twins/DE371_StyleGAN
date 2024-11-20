#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs ensemble forecast inversion using a pre-trained StyleGAN2 model.
The inversion process involves optimizing an initial random latent code so that it best represents a real ensemble forecast input.
This code use the optimization based approach and do not use the encoder based approach.

The code uses command-line arguments for setting directories, inversion parameters, and data control parameters.
The inversion is performed for a specified set of dates and lead times, generating latent code representations for real-ensemble data and saving the results.

Please make sure to configure the directory paths, parameters, and other settings based on your specific environment before running the script.

"""
import torch
import argparse
from torchvision.utils import save_image
from gan.model.stylegan2 import Generator
import os
import json
import numpy as np
import inversion.optimization_based.inversion_precip as inv
from time import perf_counter
from collections import OrderedDict
import yaml
import pandas as pd
from datetime import date, timedelta, datetime
import perturbation.utils as utils
import matplotlib.pyplot as plt


torch.manual_seed(42) #reproducibility of runs



if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################
    # Checkpoint directory - PATH to generator's weight
   #SANS EP parser.add_argument('--ckpt_dir', type = str, 
                        #default ='/scratch/mrmn/brochetc/GAN_2D/tests/Set_UseNoiseFalse/stylegan2_stylegan_dom_256_lat-dim_512_bs_8_0.002_0.002_ch-mul_2_vars_rr_u_v_t2m_noise_False/Instance_2/models/216000.pt')
    #Avec EP et noise true 
    parser.add_argument('--ckpt_dir', type = str, 
                        default ='/project/scratch/p200177/DE_371/angeliquebonamy/GAN_training/gan_training_new_dataset/exp_train_ep_with_Noise_Injection/models/102000.pt')
    # Real Data Directory - PATH to samples of the dataset
    # parser.add_argument('--real_data_dir', type = str, 
    #                     default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    parser.add_argument('--real_data_dir', type = str, 
                        default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France_rr_u_v_t2m/data/IS_rr_debug_1_1.0_0_0_0_0_0_256_large_lt/')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, 
                        default ='/project/scratch/p200177/DE_371/angeliquebonamy/results/dates/vallid_inv/inversion/')
    # Pack Directory - PATH where the packed ensembles will be saved
    parser.add_argument("--pack_dir", type=str, 
                        default = '/project/scratch/p200177/DE_371/angeliquebonamy/results/dates/vallid_inv/pack/') # storing "packed" (normalized) real data
    
    # Dataset information
    parser.add_argument("--normalization", type=str, default="minmax", choices=["minmax", "meanmax"])
    parser.add_argument('--max_file', type=str, default='max_rr_log.npy')#MaxNew_4_var.npy') # use 'MaxNew_4_var.npy' if AROME data # max_rr_log.npy
    parser.add_argument('--mean_file', type=str, default='Mean_4_var.npy') # not used if minmax normalization
    parser.add_argument('--min_file', type=str, default='min_rr_log.npy')  # not used if meanmax normalization
    
    parser.add_argument('--device', type=str, default='cuda')

    ############################ INVERSION PARAMETERS #################    

    parser.add_argument("--lr_rampup",type=float,default=0.05,help="duration of the learning rate warmup")
    parser.add_argument("--lr_rampdown",type=float, default=0.25,help="duration of the learning rate decay")
    
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    
    parser.add_argument("--noise_strength", type=float, default=0.005, help="strength of the noise level")
    parser.add_argument("--noise_ramp",type=float,default=0.75,help="duration of the noise level decay")
    
    parser.add_argument("--var_indices", type=utils.str2intlist, default=[0,1,2,3])
    parser.add_argument("--Shape", type=tuple, default=(4,256,256), help='size of the samples')
    parser.add_argument("--crop_indices", type=int, nargs='+', default=[0,256,0,256])
    
    # Progressive loss mode
    parser.add_argument("--progressive_loss_mode", type=bool, default=False, help="Progressive Loss between pixel loss and perceptual loss | Start : Only MSE | End : Only Perceptual")
    # Noise optdimization and loss noise parameter
    parser.add_argument("--noise_optimize", type=bool, default=False, help="joint optimization of noise and latent code (1) or latent code optimization only (0)?")
    parser.add_argument("--lambda_noise", type=float, default=10e6, help="weight of the noise regularization")
    # In case noise_optimize=0, the lambda_noise is not taken into account in the loss computation
    parser.add_argument("--fixed_noise", type=bool, default=False, help="Fixing the noise during optimization")

    # Parameter related to pixel loss 
    parser.add_argument('--pixel_loss_type', type=str, default='mse', choices = ['mse', 'mae','wmse','amse','wamse','sum_pixel_loss'])
    parser.add_argument("--lambda_pixel", type=float, default=10.0, help="weight of the (mae/mse/wmse) pixel loss")
    
    # Parameter related to perceptual loss 
    parser.add_argument("--lambda_vgg", type=float, default=1.0, help="weight of the vgg (perceptual) loss")
    parser.add_argument("--vgg_computation", type=str, default='sol2', choices = ['sol1', 'sol2', 'sol3', 'sol4', 'sol5'], 
                        help="Either we compute layer by layer and member per member but we have to triple th einput to make it rgb or all in one (naive)")
    parser.add_argument("--vgg_state_dict_path", type=str, default='/project/scratch/p200177/DE_371/resources/vgg_weights/vgg16-random.pth', help="Insert a path")
    #parser.add_argument("--vgg_state_dict_path", type=str, default='/project/scratch/p200177/DE_371/resources/vgg_weights/vgg16-random.pth', help="Insert a path")
    parser.add_argument("--vgg_style_layers", type=int, nargs='+', default=[], help="style layers to include in vgg loss computation")
    parser.add_argument("--vgg_feature_layers", type=int, nargs='+', default=[0,1,2,3], help="feature layers to include in vgg computation")
    parser.add_argument("--vgg_alpha_feature", type=float, default=1.0, help="weight of the feature/content loss")
    parser.add_argument("--vgg_alpha_style", type=float, default=0.01, help="weight of the style loss")
    parser.add_argument("--vgg_loss_after_step", type=float, default=0, help="compute the vgg loss only after a given number of steps")

    parser.add_argument("--invstep", type=int, default=1000, help="optimize iterations")
    parser.add_argument("--inv_checkpoints", type=utils.str2intlist, default=[250,500,1000,1500,2000])

    ########################## CONTROL of Data to invert ######################
    parser.add_argument("--dates_file", type=str, default = 'IS_boostrap_no_duplicate_rr_cumul_correct_valid.csv')
    parser.add_argument("--date_start", type=str, default = "2020-06-16")
    parser.add_argument("--date_stop", type=str, default = "2021-11-02")
    parser.add_argument("--leadtimes", type=utils.str2intlist, default=[1,3,8,12,15,19,22,25,28,31,34,37,40,45])
    
    parser.add_argument("--seed", type=int, default=42)
      # LPIPS
    parser.add_argument("--lpips_pnet", type=str, default='alex', choices=['alex','vgg','squeeze'], help="network type for lpips loss")
    parser.add_argument("--lpips_pnet_tune", action='store_true', help="tuning the weights of the pnet")
    parser.add_argument("--lpips_pnet_state_dict_path", type=str, default='/home/users/u101833/project/DE371_StyleGAN/inversion/PerceptualSimilarity/lpips/weights_pnets/alex_random.pth', help="path to lpips pre-trained network weights")
    parser.add_argument("--lambda_lpips", type=float, default=0.0, help="weight of the lpips (perceptual) loss")

    parser.add_argument("--lpips_mode", action='store_true', help="if lpips mode=False, it act like simple vgg")
    parser.add_argument("--lpips_linear_layers_state_dict_path", type=str, default='/home/users/u101833/project/DE371_StyleGAN/inversion/PerceptualSimilarity/lpips/weights_linear_layers/v0.1/vgg.pth', help="path to liunear layer lpips")
    parser.add_argument("--optimize_features_computation", action='store_true', help="Compute the features of original ensemble only once")
    parser.add_argument(
        "--noise", type=float, default=0.005, help="strength of the noise level"
    )
    parser.add_argument("--pixel_rr_vgg_others", action='store_true', help="Compute the pixel loss for rr var and vgg for u,v,t2M")

    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=10e5,
        help="weight of the noise regularization (inversion)",
    )
    params = parser.parse_args()
  

    # fix some of the inputs
    params.Shape = tuple(params.Shape)
    params.crop_indices = tuple(params.crop_indices)
    params.noise_optimize=bool(params.noise_optimize==1)

    # create output and pack directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    if not os.path.exists(params.pack_dir):
        os.makedirs(params.pack_dir)
    # set the seed for reproduciibility of runs
    seed = params.seed
    torch.manual_seed(seed)

    ################## loading dates and file names ##
    df = pd.read_csv(params.real_data_dir + params.dates_file)
    df_date = df.copy()
    df_date['Date'] = pd.to_datetime(df_date['Date'])
    df_extract = df_date[(df_date['Date']>=params.date_start) & (df_date['Date']<=params.date_stop)]

    list_dates = df_extract['Date'].unique()
    if params.normalization=="meanmax":
        Means = np.load(f'{params.real_data_dir}{params.mean_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
        Maxs = np.load(f'{params.real_data_dir}{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    #    Means = np.load(f'{params.real_data_dir}stat_files/{params.mean_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    #    Maxs = np.load(f'{params.real_data_dir}/stat_files/{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    elif params.normalization=="minmax":
    #    Mins = np.load(f'/project/scratch/p200177/DE_371/angeliquebonamy/data_basile_inv/samples_AROME_for_AE_1/stat/stat_file/{params.min_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    #    Maxs = np.load(f'/project/scratch/p200177/DE_371/angeliquebonamy/data_basile_inv/samples_AROME_for_AE_1/stat/stat_file/{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
        Mins = np.load(f'{params.real_data_dir}/stat_files_Massif_Central/{params.min_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
        Maxs = np.load(f'{params.real_data_dir}/stat_files_Massif_Central/{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    
    else:
       raise ValueError(f"Unknown normalization: {params.normalization}")

    ################ loading network #################
    G = Generator(params.Shape[1], 512,n_mlp=8,nb_var=params.Shape[0])
    ckpt = torch.load(params.ckpt_dir, map_location='cpu')['g_ema']

    if 'module' in list(ckpt.items())[0][0]: #juglling with Pytorch versioning and different module packaging
        ckpt_adapt = OrderedDict()
        for k in ckpt.keys():
            k0 = k[7:]
            ckpt_adapt[k0] = ckpt[k]
        G.load_state_dict(ckpt_adapt)
    else:
        G.load_state_dict(ckpt)

    G.eval()
    G = G.to(params.device)

    ################### producing latent mean #######
    if not os.path.exists(f'{params.output_dir}latent_mean.npy'):
        latent_z = torch.empty(10000, 512).normal_().to(params.device)
        with torch.no_grad():
            w = G.style(latent_z)
        latent_mean = w.mean(dim=0).detach().cpu()
        np.save(f'{params.output_dir}latent_mean.npy',latent_mean.numpy())
    else : 
        lm = np.load(f'{params.output_dir}latent_mean.npy').astype(np.float32)
        latent_mean = torch.tensor(lm, dtype = torch.float32)

    ########### write inversion parameters to file ############
    config_file = params.output_dir + "inversion_params.yaml"
    print("writing params config file:", config_file)
    try:
        file=open(config_file,"w")
        yaml.dump(params.__dict__,file)
    except Exception as e:
         print("unable to write params config file")
         print(e)

    # print inversion parameters
    print("\nInversion parameters:")
    for key, value in params.__dict__.items():
        print(f"{key}: {value}")

    if params.vgg_loss_after_step >= params.invstep:
        print('The parameters vgg_loss_after_step cannot be superior or equal to the number of optim steps')
        raise ValueError
    # print(list_dates)
    # for i,j in enumerate(os.listdir('./datas_clement')):
    #     print('JJJJJ',j)
    #     Ens_r = torch.from_numpy(np.load(f'./datas_clement/{j}').astype(np.float32))
    #     # Ens_r = torch.tensor(-1. + 2*(Ens_r - Mins) / (Maxs-Mins), dtype = torch.float32)
    #     np.save(params.pack_dir+f'{j}', Ens_r.numpy().astype(np.float32))
    #     inv.optimize(Ens_r, G, latent_mean, params.device, params,j)
# #PLOT GENERATOR ::
#     device = params.device if torch.cuda.is_available() else 'cpu'

#     nber_imgs = 10
#     latent_z = torch.empty(nber_imgs, 512).normal_().to(device)  # Single latent vector        
#     with torch.no_grad():
#         styles = G.style(latent_z)  # Get styles
#         print("Shape of latent_z:", latent_z.shape)
#         print("Shape of styles:", styles.shape)
#         generated_image = G([styles])  # Generate image
#     print(f"Length of generated_image tuple: {len(generated_image)}")
#     for i, img in enumerate(generated_image):
#         print(f"Element {i} type: {type(img)}")
#         if isinstance(img, torch.Tensor):
#             print(f"Element {i} shape: {img.shape}")
#     # Extract the image tensor from the tuple

#     image_tensor = generated_image[0]
#     print(f"Shape of image_tensor: {image_tensor.shape}")


#         # Remove the batch dimension
#     image_tensor = image_tensor.squeeze(0)  # Shape is now [4, 256, 256]

#         # Convert the tensor to numpy for plotting
#     image_np = image_tensor.detach().cpu().numpy()

#         # Set up the figure with 4 subplots (one for each variable)
#     fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns

#         # Loop over the 4 channels and plot each one
#     for j in range(nber_imgs):
#         for i in range(4):
#             if i==3:
#                 axs[i].imshow(image_np[j][i], cmap='coolwarm',origin='lower')  # Plot in grayscale (assuming each variable is grayscale)
#                 axs[i].axis('off')  # Turn off the axis
#                 axs[i].set_title(f'Variable {i+1}')  # Set title for each variable
#                 fig.savefig(f'{os.path.join(params.output_dir)}/generated_image_{j}.png')
#             else:
                    
#                 axs[i].imshow(image_np[j][i], cmap='viridis',origin='lower')  # Plot in grayscale (assuming each variable is grayscale)
#                 axs[i].axis('off')  # Turn off the axis
#                 axs[i].set_title(f'Variable {i+1}')  # Set title for each variable
#                 fig.savefig(f'{os.path.join(params.output_dir)}/generated_image_{j}.png')


# End plots generator 
    ################## main loop ##################
    for date_ in list_dates:
        print(date_)
        print((df_extract['Date']==date_).sum())
        datename = date_.strftime('%Y-%m-%d')
        print("\n===========================")
        for lt in params.leadtimes:
            params.date_index = datename
            params.lt_index = lt
            
            
            # Check if the files already exists (to qave computation time)
            already_exist = []
            if os.path.isfile(params.pack_dir+f'Rsemble_{datename}_{lt}.npy'):
                already_exist.append(True)
            else :
                already_exist.append(False)
            for i in params.inv_checkpoints :
                if os.path.isfile(params.output_dir+'w_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i)):
                    already_exist.append(True)
                else :
                    already_exist.append(False)
                if os.path.isfile(params.output_dir+'invertFsemble_{}_{}_{}.npy'.format(params.date_index,params.lt_index,i)):
                    already_exist.append(True)
                else :
                    already_exist.append(False)
                if os.path.isfile(params.output_dir+'noise_{}_{}_{}.p'.format(params.date_index,params.lt_index,i)):
                    already_exist.append(True)
                else :
                    already_exist.append(False)

            if np.all(already_exist) :
                print('The inversion was already done for the date {} with leadtime {}. This sample is skipped.'.format(datename,lt))
            else :
                print('Launching inversion process for the date {} with leadtime {}.'.format(datename,lt))
                df_extract = df_extract.rename(columns={'Leadtime':'LeadTime'})
                df0 = df_extract[(df_extract['Date']==date_) & (df_extract['LeadTime']==lt-1)]
                if len(df0)==0:
                   print("# samples: 0")
                   continue
                print('JE SUIS LE CSV')
                Ens_r = utils.load_batch_from_timestamp(df_extract, date_, lt-1, params.real_data_dir, Shape=params.Shape, var_indices=params.var_indices) #, crop_indices=params.crop_indices)
                print('JE SUIS LE TYPE',type(Ens_r),Ens_r.shape)
                # n_samples = np.min([Ens_r.shape[0], 6])
                # print(f"extracting {n_samples} samples for inversion\n")
                # Ens_r = Ens_r[:n_samples]
                #log pour les précipitations
                channel_rr=Ens_r[:,0,:,:]
                transformed_channel_rr = np.log(1+channel_rr)
                Ens_r[:,0,:,:]=transformed_channel_rr
                # normalise samples and save in pack dir. obs! make sure normalization is done correctly (according to how model was trained)
                if params.normalization=="meanmax":
                   Ens_r = torch.tensor(0.95*(Ens_r - Means) / (Maxs), dtype = torch.float32)
                elif params.normalization=="minmax":
                   print('NORMALLLLL',Mins,Maxs)
                   Ens_r = torch.tensor(-1. + 2*(Ens_r - Mins) / (Maxs-Mins), dtype = torch.float32)
                else:
                   raise ValueError(f"Unknown normalization: {params.normalization}")
                
                np.save(params.pack_dir+f'Rsemble_{datename}_{lt}.npy', Ens_r.numpy().astype(np.float32))
                #print('JE SUIS DANS LE MAIN ',  np.shape(Ens_r[0][1].cpu().detach().numpy()),Ens_r[0][1].cpu().detach().numpy().astype(float))
                print('PARAMETRES1', params)
                #Ens_r = torch.from_numpy(np.load(f'j').astype(np.float32))
                # Ens_r = torch.tensor(-1. + 2*(Ens_r - Mins) / (Maxs-Mins), dtype = torch.float32)
                #np.save(params.pack_dir+f'j', Ens_r.numpy().astype(np.float32))
                # inv.optimize(Ens_r, G, latent_mean, params.device, params)
                batch_dir = ''
                batch_idx= 1
                scenario = 'inversion'
                inv.optimize(batch_dir,Ens_r,batch_idx, G, latent_mean, params.device, params,scenario)










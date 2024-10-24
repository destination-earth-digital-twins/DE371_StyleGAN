#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import os
import argparse

from gan.model.stylegan2 import Generator
from collections import OrderedDict
import gan.data.dataset_handler_ddp as DSH
from torch.utils.data import DataLoader
import numpy as np
from main_gan import str2list, str2bool, str2intlist, str2inttuple
import matplotlib.pyplot as plt
import scipy


if __name__=="__main__" :
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument('--data_dir', type=str, default="/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/")
    parser.add_argument('--mean_file', type=str, default=None )
    parser.add_argument('--std_file', type=str, default=None )
    parser.add_argument('--max_file', type=str, default=None )
    parser.add_argument('--min_file', type=str, default=None )
    parser.add_argument('--id_file', type=str, default="Large_lt_train_labels_1.csv")
    parser.add_argument('--pretrained_model', type=int, default=-1)
    parser.add_argument('--training_dir', type=str, default='/project/home/p200177/DE_371/experiments_WP1/gan_training/exp5/')
    parser.add_argument('--training_step', type=int, default=[106000,206000])
    # Model architecture hyper-parameters
    
    parser.add_argument('--model', type=str, default='stylegan2', \
                        choices=['stylegan2', 'stylegan2_fp16'])
    
    # choices of loss function and initialization
    parser.add_argument('--train_type', type=str, default='stylegan',\
                        choices=['stylegan', 'wave_d'])
    #architectural choices
    
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--g_channels', type=int, default=45)
    parser.add_argument('--d_channels', type=int, default=45)
    parser.add_argument('--n_mlp', type=int, default=8, help="depth of the z->w mlp")
    parser.add_argument("--channel_multiplier",type=int, default=2,
        help="channel multiplier factor for the stylegan/swagan model. config-f = 2, else = 1",
    )

    parser.add_argument("--tanh_output", type=str2bool, default=False, help="if True, add tanh non linearity before Generator output")

    # regularisation settings (styleGAN)
    
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--path_regularize",type=float,default=2,\
                        help="weight of the path length regularization")

    parser.add_argument( "--path_batch_shrink",type=int,default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)")
    
    parser.add_argument("--d_reg_every",type=int,default=16,
                        help="interval of the applying r1 regularization")
    
    parser.add_argument("--g_reg_every",type=int, default=4,
        help="interval of the applying path length regularization")
    
    parser.add_argument("--mixing", type=float, default=0.9, 
                        help="probability of latent code mixing")
    
    # augmentation and ADA settings (styleGAN)
    
    parser.add_argument("--augment", action="store_true", 
                        help="apply non leaking augmentation"
    )
    parser.add_argument("--augment_p", type=float, default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument("--ada_target",type=float,default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument("--ada_length",type=int, default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument("--ada_every", type=int,default=256,
                        help="probability update interval of the adaptive augmentation",
    )

    # Training settings
    parser.add_argument('--epochs_num', type=int, default=30,\
                        help='how many times to go through dataset')
    parser.add_argument('--total_steps', type=int, default=500001,\
                        help='how many times to update the generator')
    
    parser.add_argument('--batch_size', type=int, default=8)

    
    parser.add_argument('--lr_G', type=float, default=0.002)
    parser.add_argument('--lr_D', type=float, default=0.002)
    
    parser.add_argument('--beta1_D', type=float, default=0.0)
    parser.add_argument('--beta2_D', type=float, default=0.9)
    
    parser.add_argument('--beta1_G', type=float, default=0.0)
    parser.add_argument('--beta2_G', type=float, default=0.9)
    
    parser.add_argument('--warmup', type=str2bool, default=False)
    parser.add_argument('--use_noise', type=str2bool, default=False, help="if False, doesn't use noise_inj")
    
    # Data description
    parser.add_argument('--var_names', type=str2list, default=['u','v','t2m'])#, 'orog'])
    parser.add_argument('--crop_indexes', type=str2intlist, default=[0,256,0,256])

    parser.add_argument('--crop_size', type=str2inttuple, default=(256,256) ) #   if not all_domain else (256,256))
    parser.add_argument('--full_size', type=str2inttuple, default=(256,256))
    
    # Data Description - Temporal Aspect
    parser.add_argument('--multi_timestep_mode', action='store_true')
    parser.add_argument('--timestep_labelling', action='store_true')
    parser.add_argument('--variable_first', action='store_true')
    parser.add_argument('--nb_timesteps', type=int, default=15)
    parser.add_argument('--timestep_period', type=int, default=3)
    parser.add_argument('--stack_sample_along_time_and_variable', action='store_true')
    parser.add_argument('--cutoff_dataset_leadtimes', action='store_true', help='To only consider [t+dt, t+2*dt...] and not the leadtime between t and dt')
    
    # Training settings -schedulers
    parser.add_argument('--lrD_sched', type=str, default='None', \
                        choices=['None','exp', 'linear', 'cyclic'])
    parser.add_argument('--lrG_sched', type=str, default='None', \
                        choices=['None','exp', 'linear', 'cyclic'])
    parser.add_argument('--lrD_gamma', type=float, default=0.95)
    parser.add_argument('--lrG_gamma', type=float, default=0.95)
    
    
    # Testing and plotting setting
    parser.add_argument('--test_samples',type=int, default=16 ) # if all_domain else 256,help='samples to be tested')
    parser.add_argument('--plot_samples', type=int, default=16)
    parser.add_argument('--sample_num', type=int, default=16, help='Samples to be saved') #  if all_domain else 256,\
                        

    # Misc
    parser.add_argument('--fp16_resolution', type=int, default=1000) # 1000 --> not used
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

    # Step size

    parser.add_argument('--log_epoch', type=int, default=1000)
    # parser.add_argument('--sample_epoch', type=int, default=0)
    parser.add_argument('--plot_epoch', type=int, default=1000)
    parser.add_argument('--save_epoch', type=int, default=1000)
    parser.add_argument('--test_epoch', type=int, default=1000)
    parser.add_argument('--save_step', type=int, default=2000)# if very_small_exp else (1000 if small_exp else 3000)) # set to 0 if not needed

    # Not used in trainer_ddp
    parser.add_argument('--log_step', type=int, default=2000)# if very_small_exp else (1000 if small_exp else 3000)) #-> default is at the end of each epoch
    parser.add_argument('--sample_step', type=int, default=2000)# if very_small_exp else (1000 if small_exp else 3000)) # set to 0 if not needed
    parser.add_argument('--plot_step', type=int, default=2000)# if very_small_exp else (1000 if small_exp else 3000)) #set to 0 if not needed
    parser.add_argument('--test_step', type=int, default=2000)# if very_small_exp else (1000 if small_exp else 3000)) #set to 0 if not needed

    # parser.add_argument('--confi/home/mrmn/sanchezv/project/code/styleganpnria/gan/configs/Set_UseNoiseFalseg_dir', type=str, default="/home/users/u101833/project/DE371_StyleGAN/gan/configs/Set_UseNoiseFalse/", help="The config files absolute path")
    parser.add_argument('--config_dir', type=str, default="/project/home/p200177/DE_371/experiments_WP1/gan_training/Set_UseNoiseFalse/", help="The config files absolute path")
    parser.add_argument('--dataset_handler_config', type=str, default="dataset_handler_config.yaml", help="The dataset_handler config file")
    parser.add_argument('--scheduler_config', type=str, default="scheduler_config.yaml", help="The scheduler config file")
    print("instantiating dataset")
    config = parser.parse_args()
    config.multi_timestep_mode=True
    config.cutoff_dataset_leadtimes=True

    # Setting coordinates of special points on map
    alpes_mountain_coord = [125,240]
    pyrenean_mountain_coord = [25,20]
    massif_central_mountain_coord = [110,128]
    sea_coord = [30,140]
    toulouse_coord = [63,38]

    pixel_coordinate_dict = {'toulouse' : toulouse_coord,
                        'alpes' : alpes_mountain_coord,
                        'pyrenee' : pyrenean_mountain_coord,
                        'massif_central': massif_central_mountain_coord,
                        'sea' : sea_coord
                        }

    print('pixel_coordinate:', pixel_coordinate_dict)

    generators = []
    for training_step in config.training_step:

        print(f"instantiating generator for step {training_step}")
        G = Generator(256, 512, n_mlp=8, nb_var=config.g_channels, channel_multiplier=config.channel_multiplier)
        
        ckpt = torch.load(config.training_dir+f'models/{training_step}.pt', map_location='cpu')['g_ema']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("using device:", device)

        if 'module' in list(ckpt.items())[0][0]: # juglling with Pytorch versioning and different module packaging
            ckpt_adapt = OrderedDict()
            for k in ckpt.keys():
                k0 = k[7:]
                ckpt_adapt[k0] = ckpt[k]
            G.load_state_dict(ckpt_adapt)
        else:
            G.load_state_dict(ckpt)

        G.eval()
        G = G.to(device)
        generators.append(G)

    output_dir = config.training_dir+'scores/plots/diurnal_cycle/'
    
    
    # Dataset loading
    Dl_train = DSH.ISData_Loader("Train", config)

    dataset = DSH.ISDataset(
        config, 
        Dl_train.dataset_handler_yaml, 
        'coords', 
        variable_indices=[1,2,3], 
        transform=Dl_train.transform(), 
        detransform=Dl_train.detransform()
    )

    train_dataloader = DataLoader(dataset = dataset,
                        batch_size = 16,
                        shuffle = False,
                        drop_last = True,
                        num_workers=1
    )
    nb_batch =16
    nb_sample_total = nb_batch * 16

    print(f'Eval done on {nb_sample_total} samples')

    diurnal_cycle = np.zeros((len(generators)+1, len(pixel_coordinate_dict), 2, config.nb_timesteps, nb_sample_total))
    pearsons_first_to_each_leadtime_img = np.zeros((len(generators)+1, 3, config.nb_timesteps, nb_sample_total))
    pearsons_sliding_img = np.zeros((len(generators)+1, 3, config.nb_timesteps-1, nb_sample_total))
    temporal_difference = np.zeros((len(generators)+1, 3, config.nb_timesteps-1, nb_sample_total))

    loop = enumerate(train_dataloader)
    cursor = 0
    
    for i, batch in loop:
        if i > nb_batch-1 : 
            break
        # Computing diurnal cycle for original samples
        
        img, _, _ = batch
        sample = img.numpy()
        for key_id, key in enumerate(pixel_coordinate_dict):
            pixel_coordinate=pixel_coordinate_dict[key]
            for member_id in range(len(img)):
                real_sample = sample[member_id].transpose(2,3,1,0)
                real_sample = np.array([dataset.detransform(real_sample[:,:,:,t]) for t in range(config.nb_timesteps)])
                for t in range(config.nb_timesteps):
                    _sample = real_sample[t]

                    # Diurnal Cycle
                    u = _sample[0][pixel_coordinate[0]][pixel_coordinate[1]]
                    v = _sample[1][pixel_coordinate[0]][pixel_coordinate[1]]
                    t2m = _sample[2][pixel_coordinate[0]][pixel_coordinate[1]]
                    diurnal_cycle[0, key_id, 0, t, cursor+member_id] = np.sqrt(u**2+v**2)
                    diurnal_cycle[0, key_id, 1, t, cursor+member_id] = t2m
                    
                    
                    for var_id in range(3):
                        # Pearson Correlation on generated samples
                        pearsons_first_to_each_leadtime_img[0, var_id, t, cursor+member_id] = scipy.stats.pearsonr(
                                                                        real_sample[0][var_id].flatten(),
                                                                        real_sample[t][var_id].flatten()
                        ).statistic
                        if t==0 :
                            pearsons_sliding_img[0, var_id, t, cursor+member_id]=np.nan
                        elif t < config.nb_timesteps-1:
                            pearsons_sliding_img[0, var_id, t, cursor+member_id] = scipy.stats.pearsonr(
                                                            real_sample[t][var_id].flatten(),
                                                            real_sample[t+1][var_id].flatten()
                            ).statistic

                        # Temporal Difference
                        if t==0 :
                            temporal_difference[0, var_id, t, cursor+member_id]=np.nan
                        elif t < config.nb_timesteps-1:
                            temporal_difference[0, var_id, t, cursor+member_id] = np.mean(np.abs(real_sample[t+1][var_id] - real_sample[t][var_id]))

                for checkpoint_id in range(1, len(config.training_step)+1):
                    z = torch.empty(1, 512).normal_().to(device)

                    with torch.no_grad():
                        gen_sample, _, _ = generators[checkpoint_id-1]([z])
                        gen_sample = gen_sample.cpu().numpy()
                        gen_sample = gen_sample.reshape((config.nb_timesteps, len(config.var_names), np.shape(sample)[-2], np.shape(sample)[-1]))
                        gen_sample = gen_sample.transpose(2,3,1,0)
                        gen_sample = np.array([dataset.detransform(gen_sample[:,:,:,t]) for t in range(config.nb_timesteps)])
                    for t in range(config.nb_timesteps):
                        _sample= gen_sample[t]

                        # Diurnal Cycle
                        u = _sample[0][pixel_coordinate[0]][pixel_coordinate[1]]
                        v = _sample[1][pixel_coordinate[0]][pixel_coordinate[1]]
                        diurnal_cycle[checkpoint_id, key_id, 0, t, cursor+member_id] = np.sqrt(u**2+v**2)
                        diurnal_cycle[checkpoint_id, key_id, 1, t, cursor+member_id] = _sample[2][pixel_coordinate[0]][pixel_coordinate[1]]

                        # Pearson Correlation on generated samples
                        for var_id in range(3):
                            pearsons_first_to_each_leadtime_img[checkpoint_id, var_id, t, cursor+member_id] = scipy.stats.pearsonr(
                                                                            gen_sample[0][var_id].flatten(),
                                                                            gen_sample[t][var_id].flatten()
                            ).statistic
                            if t==0 :
                                pearsons_sliding_img[checkpoint_id, var_id, t, cursor+member_id]=np.nan
                            elif t < config.nb_timesteps-1:
                                pearsons_sliding_img[checkpoint_id, var_id, t, cursor+member_id] = scipy.stats.pearsonr(
                                                                gen_sample[t][var_id].flatten(),
                                                                gen_sample[t+1][var_id].flatten()
                                ).statistic

                            # Temporal Difference
                            if t==0 :
                                temporal_difference[checkpoint_id, var_id, t, cursor+member_id]=np.nan
                            elif t < config.nb_timesteps-1:
                                temporal_difference[checkpoint_id, var_id, t, cursor+member_id] = np.mean(np.abs(gen_sample[t+1][var_id] - gen_sample[t][var_id]))


        cursor+=16

    list_ticks = np.arange(0, 45, config.timestep_period)

    diurnal_cycle = np.mean(diurnal_cycle, -1)
    print('Saving Diurnal Cycle')
    np.save(output_dir+f'Diurnal_Cycle_over_{nb_sample_total}_samples_{config.nb_timesteps}_nb_var_{len(config.var_names)}.npy', diurnal_cycle)
    print('Plotting Diurnal Cycle')
    for key_id, key in enumerate(pixel_coordinate_dict):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16,16))
        ax[0].plot(range(config.nb_timesteps),  diurnal_cycle[0,key_id,0], linewidth=6, color='k', label='AROME')
        ax[0].set_ylabel('Wind speed (m/s)', size = 30)
        ax[0].set_xticks(range(15), labels=list_ticks)

        ax[1].plot(range(config.nb_timesteps), diurnal_cycle[0,key_id,1], linewidth=6, color='k', label='AROME')
        ax[1].set_ylabel('Temperature at 2m (K)', size = 30)
        ax[1].set_xticks(range(15), labels=list_ticks)

        for checkpoint_id in range(1, len(config.training_step)+1):
            ax[0].plot(range(config.nb_timesteps), diurnal_cycle[checkpoint_id,key_id,0], linewidth=6, label=f'Generated - {config.training_step[checkpoint_id-1]}')
            ax[0].legend(prop={'size':20})
            ax[1].plot(range(config.nb_timesteps), diurnal_cycle[checkpoint_id,key_id,1], linewidth=6, label=f'Generated - {config.training_step[checkpoint_id-1]}')
            ax[1].legend(prop={'size':20})

        fig.suptitle(f'Diurnal Cycle on pixel {key}', size=30)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(output_dir+f'Diurnal_Cycle_over_{nb_sample_total}_samples_{config.nb_timesteps}_nb_var_{len(config.var_names)}_{key}.png') 
    
    
    pearsons_sliding_img = np.mean(pearsons_sliding_img, -1)
    pearsons_first_to_each_leadtime_img = np.mean(pearsons_first_to_each_leadtime_img, -1)
    
    print('Saving Pearson Correlation')
    np.save(output_dir+f'Pearson_Correlation_first_to_each_over_{nb_sample_total}_samples_{config.nb_timesteps}_nb_var_{len(config.var_names)}.npy', pearsons_first_to_each_leadtime_img)
    np.save(output_dir+f'Pearson_Correlation_sliding_over_{nb_sample_total}_samples_{config.nb_timesteps}_nb_var_{len(config.var_names)}.npy', pearsons_sliding_img)
    print('Plotting Pearson Correlation')
    
    
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16,16))
    ax[0].plot(range(config.nb_timesteps),  pearsons_first_to_each_leadtime_img[0, 0], linewidth=6, color='k', label='AROME')
    ax[0].set_ylabel('Wind speed U (m/s)', size = 30)
    ax[0].set_xticks(range(15), labels=list_ticks)
    
    ax[1].plot(range(config.nb_timesteps),  pearsons_first_to_each_leadtime_img[0, 1], linewidth=6, color='k', label='AROME')
    ax[1].set_ylabel('Wind speed V (m/s)', size = 30)
    ax[1].set_xticks(range(15), labels=list_ticks)

    ax[2].plot(range(config.nb_timesteps), pearsons_first_to_each_leadtime_img[0, 2], linewidth=6, color='k', label='AROME')
    ax[2].set_ylabel('Temperature at 2m (K)', size = 30)
    ax[2].set_xticks(range(15), labels=list_ticks)

    for checkpoint_id in range(1, len(config.training_step)+1):
        ax[0].plot(range(config.nb_timesteps), pearsons_first_to_each_leadtime_img[checkpoint_id, 0], linewidth=6, label=f'Generated - {config.training_step[checkpoint_id-1]}')
        ax[0].legend(prop={'size':20})
        ax[1].plot(range(config.nb_timesteps), pearsons_first_to_each_leadtime_img[checkpoint_id, 1], linewidth=6, label=f'Generated - {config.training_step[checkpoint_id-1]}')
        ax[1].legend(prop={'size':20})
        ax[2].plot(range(config.nb_timesteps), pearsons_first_to_each_leadtime_img[checkpoint_id, 2], linewidth=6, label=f'Generated - {config.training_step[checkpoint_id-1]}')
        ax[2].legend(prop={'size':20})

    fig.suptitle('Pearson Correlation First to Each Leadtime', size=30)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(output_dir+f'Pearson_Correlation_first_to_each_over_{nb_sample_total}_samples_{config.nb_timesteps}_nb_var_{len(config.var_names)}.png') 

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16,16))
    ax[0].plot(range(config.nb_timesteps-1),  pearsons_sliding_img[0, 0], linewidth=6, color='k', label='AROME')
    ax[0].set_ylabel('Wind speed U (m/s)', size = 30)
    ax[0].set_xticks(range(15), labels=list_ticks)

    ax[1].plot(range(config.nb_timesteps-1),  pearsons_sliding_img[0, 1], linewidth=6, color='k', label='AROME')
    ax[1].set_ylabel('Wind speed V (m/s)', size = 30)
    ax[1].set_xticks(range(15), labels=list_ticks)
    
    ax[2].plot(range(config.nb_timesteps-1), pearsons_sliding_img[0, 2], linewidth=6, color='k', label='AROME')
    ax[2].set_ylabel('Temperature at 2m (K)', size = 30)
    ax[2].set_xticks(range(15), labels=list_ticks)

    for checkpoint_id in range(1, len(config.training_step)+1):
        ax[0].plot(range(config.nb_timesteps-1), pearsons_sliding_img[checkpoint_id, 0], linewidth=6, label=f'Generated - {config.training_step[checkpoint_id-1]}')
        ax[0].legend(prop={'size':20})
        ax[1].plot(range(config.nb_timesteps-1), pearsons_sliding_img[checkpoint_id, 1], linewidth=6, label=f'Generated - {config.training_step[checkpoint_id-1]}')
        ax[1].legend(prop={'size':20})
        ax[2].plot(range(config.nb_timesteps-1), pearsons_sliding_img[checkpoint_id, 2], linewidth=6, label=f'Generated - {config.training_step[checkpoint_id-1]}')
        ax[2].legend(prop={'size':20})

    fig.suptitle('Pearson Correlation between X(t) and X(t+1)', size=30)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(output_dir+f'Pearson_Correlation_sliding_over_{nb_sample_total}_samples_{config.nb_timesteps}_nb_var_{len(config.var_names)}.png') 


    temporal_difference = np.mean(temporal_difference, -1)

    print('Saving Temporal Difference')
    np.save(output_dir+f'Temporal_Difference_over_{nb_sample_total}_samples_{config.nb_timesteps}_nb_var_{len(config.var_names)}.npy', temporal_difference)
    print('Plotting Temporal Difference')

    
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16,16))
    ax[0].plot(range(config.nb_timesteps-1),  temporal_difference[0, 0], linewidth=6, color='k', label='AROME')
    ax[0].set_ylabel('Wind speed U (m/s)', size = 30)
    ax[0].set_xticks(range(15), labels=list_ticks)

    ax[1].plot(range(config.nb_timesteps-1),  temporal_difference[0, 1], linewidth=6, color='k', label='AROME')
    ax[1].set_ylabel('Wind speed V (m/s)', size = 30)
    ax[1].set_xticks(range(15), labels=list_ticks)

    ax[2].plot(range(config.nb_timesteps-1), temporal_difference[0, 2], linewidth=6, color='k', label='AROME')
    ax[2].set_ylabel('Temperature at 2m (K)', size = 30)
    ax[2].set_xticks(range(15), labels=list_ticks)

    for checkpoint_id in range(1, len(config.training_step)+1):
        ax[0].plot(range(config.nb_timesteps-1), temporal_difference[checkpoint_id, 0], linewidth=6, label=f'Generated - {config.training_step[checkpoint_id-1]}')
        ax[0].legend(prop={'size':20})
        ax[1].plot(range(config.nb_timesteps-1), temporal_difference[checkpoint_id, 1], linewidth=6, label=f'Generated - {config.training_step[checkpoint_id-1]}')
        ax[1].legend(prop={'size':20})
        ax[2].plot(range(config.nb_timesteps-1), temporal_difference[checkpoint_id, 2], linewidth=6, label=f'Generated - {config.training_step[checkpoint_id-1]}')
        ax[2].legend(prop={'size':20})

    fig.suptitle('Temporal Difference for Each Leadtime : ∆X = |X(t+1) - X(t)|', size=30)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(output_dir+f'Temporal_Difference_over_{nb_sample_total}_samples_{config.nb_timesteps}_nb_var_{len(config.var_names)}.png') 

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
from main_gan import get_expe_parameters
import matplotlib.pyplot as plt
import scipy
from tqdm import trange

print("instantiating dataset")
config = get_expe_parameters().parse_args()
config.stat_folder = ''
config.crop_indexes = [0,256,0,256]
config.crop_size = (256,256)
config.multi_timestep_mode = True
config.nb_timesteps = 8
config.timestep_period = 6
config.stack_sample_along_time_and_variable = False
config.var_names=['u','v','t2m']


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

real_only = False
output_dir = '/project/scratch/p200177/DE_371/victorsanchez/results/gan_training/exp11_seq_GAN_exp_train_sequential_every_6h_10000_u_v_t2m_channel_multiplier=6/scores/plots/diurnal_cycle/'
if not real_only:
    print("instantiating generator")
    G = Generator(256, 512, n_mlp=8, nb_var=config.nb_timesteps*len(config.var_names), channel_multiplier=6)
    training_step = 68000
    ckpt = torch.load(f'/project/scratch/p200177/DE_371/victorsanchez/results/gan_training/exp11_seq_GAN_exp_train_sequential_every_6h_10000_u_v_t2m_channel_multiplier=6/models/0{training_step}.pt', map_location='cpu')['g_ema']
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


for key in pixel_coordinate_dict:
    pixel_coordinate=pixel_coordinate_dict[key]
    print('computing diurnal cycle on zone :', key)
    nb_sample_total = 230
    value_at_given_pixel_per_variable_real_sample_t2m = []
    value_at_given_pixel_per_variable_generated_sample_t2m = []
    if 'u' in config.var_names:
        value_at_given_pixel_per_variable_real_sample_wind = []
        value_at_given_pixel_per_variable_generated_sample_wind = []
    
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
                        num_workers=1)

    loop = enumerate(train_dataloader)

    for i, batch in loop:
        if i > nb_sample_total : 
            break
        # Computing diurnal cycle for original samples
        img, _, _ = batch
        sample = img.numpy()

        if not real_only:    
            # Computing diurnal cycle for generated samples
            z = torch.empty(16, 512).normal_().to(device)

            with torch.no_grad():
                gen_sample, _, _ = G([z])
                gen_sample = gen_sample.cpu().numpy()
                gen_sample = gen_sample.reshape((16,config.nb_timesteps, len(config.var_names), np.shape(sample)[-2], np.shape(sample)[-1]))
        # for var_id in range(len(config.var_names)):
        value_at_given_pixel_real_sample_t2m=[]
        if not real_only:
            value_at_given_pixel_generated_sample_t2m=[]
        if 'u' in config.var_names:
            value_at_given_pixel_real_sample_wind=[]
            if not real_only:
                value_at_given_pixel_generated_sample_wind=[]
        for t in range(len(img[0])):
            value_at_t_real_sample_t2m = []
            if not real_only:
                value_at_t_generated_sample_t2m = []
            if 'u' in config.var_names:
                value_at_t_real_sample_wind = []
                if not real_only:
                    value_at_t_generated_sample_wind = []
            for member_id in range(len(img)):
                # denorm original sample
                # _sample = sample[member_id][t] # normalized 
                # print('sample stat u',np.min(_sample[0]), np.mean(_sample[0]), np.max(_sample[0]))
                # print('sample stat v',np.min(_sample[1]), np.mean(_sample[1]), np.max(_sample[1]))
                # print('sample stat t2m',np.min(_sample[2]), np.mean(_sample[2]), np.max(_sample[2]))
                _sample = dataset.detransform(sample[member_id][t].transpose((1,2,0))).numpy() # denormalized 
                # print('shape real',np.shape(_sample))
                if 'u' in config.var_names: 
                    value_at_t_real_sample_t2m.append(_sample[2][pixel_coordinate[0]][pixel_coordinate[1]])
                    u = _sample[0][pixel_coordinate[0]][pixel_coordinate[1]]
                    v = _sample[1][pixel_coordinate[0]][pixel_coordinate[1]]
                    value_at_t_real_sample_wind.append(np.sqrt(u**2+v**2))
                else :
                    value_at_t_real_sample_t2m.append(_sample[2][pixel_coordinate[0]][pixel_coordinate[1]])
                if not real_only:
                    # denorm gen sample
                    # _sample = gen_sample[member_id][t] # normalized 
                    # print('gen sample stat u',np.min(_sample[0]), np.mean(_sample[0]), np.max(_sample[0]))
                    # print('gen sample stat v',np.min(_sample[1]), np.mean(_sample[1]), np.max(_sample[1]))
                    # print('gen sample stat t2m',np.min(_sample[2]), np.mean(_sample[2]), np.max(_sample[2]))
                    _sample = dataset.detransform(gen_sample[member_id][t].transpose((1,2,0))).numpy() # denormalized 
                    # print('shape gen',np.shape(_sample))
                    if 'u' in config.var_names: 
                        value_at_t_generated_sample_t2m.append(_sample[2][pixel_coordinate[0]][pixel_coordinate[1]])
                        u = _sample[0][pixel_coordinate[0]][pixel_coordinate[1]]
                        v = _sample[1][pixel_coordinate[0]][pixel_coordinate[1]]
                        value_at_t_generated_sample_wind.append(np.sqrt(u**2+v**2))
                    else :
                        value_at_t_generated_sample_t2m.append(_sample[0][pixel_coordinate[0]][pixel_coordinate[1]])

            value_at_given_pixel_real_sample_t2m.append(np.mean(value_at_t_real_sample_t2m))
            if not real_only:
                value_at_given_pixel_generated_sample_t2m.append(np.mean(value_at_t_generated_sample_t2m))
            if 'u' in config.var_names: 
                value_at_given_pixel_real_sample_wind.append(np.mean(value_at_t_real_sample_wind))
                if not real_only:
                    value_at_given_pixel_generated_sample_wind.append(np.mean(value_at_t_generated_sample_wind))

        value_at_given_pixel_per_variable_real_sample_t2m.append(value_at_given_pixel_real_sample_t2m)
        if not real_only:
            value_at_given_pixel_per_variable_generated_sample_t2m.append(value_at_given_pixel_generated_sample_t2m)
        if 'u' in config.var_names: 
            value_at_given_pixel_per_variable_real_sample_wind.append(value_at_given_pixel_real_sample_wind)
            if not real_only:
                value_at_given_pixel_per_variable_generated_sample_wind.append(value_at_given_pixel_generated_sample_wind)

        

    # Computing averages
    # print(np.shape(value_at_given_pixel_per_variable_real_sample_t2m))
    value_at_given_pixel_per_variable_real_sample_t2m = np.mean(value_at_given_pixel_per_variable_real_sample_t2m, axis=0)
    # print(np.shape(value_at_given_pixel_per_variable_real_sample_t2m))
    if not real_only:
        value_at_given_pixel_per_variable_generated_sample_t2m = np.mean(value_at_given_pixel_per_variable_generated_sample_t2m, axis=0)
    if 'u' in config.var_names: 
        value_at_given_pixel_per_variable_real_sample_wind = np.mean(value_at_given_pixel_per_variable_real_sample_wind, axis=0)
        if not real_only:
            value_at_given_pixel_per_variable_generated_sample_wind = np.mean(value_at_given_pixel_per_variable_generated_sample_wind, axis=0)

    if 'u' in config.var_names: 
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16,16))
        ax[0].plot(range(config.nb_timesteps), value_at_given_pixel_per_variable_real_sample_t2m, linewidth=6, color='k', label='AROME')
        ax[0].set_ylabel('Temperature at 2m (K)', size = 30)
        if not real_only:
            ax[0].plot(range(config.nb_timesteps), value_at_given_pixel_per_variable_generated_sample_t2m, linewidth=6, color='tab:orange', label='Generated')
        ax[0].legend(prop={'size':20})
        ax[1].plot(range(config.nb_timesteps), value_at_given_pixel_per_variable_real_sample_wind, linewidth=6, color='k', label='AROME')
        ax[1].set_ylabel('Wind speed (m/s)', size = 30)
        if not real_only:
            ax[1].plot(range(config.nb_timesteps), value_at_given_pixel_per_variable_generated_sample_wind, linewidth=6, color='tab:orange', label='Generated')
        ax[1].legend(prop={'size':20})
        fig.suptitle(f'Diurnal Cycle on pixel {key}', size=30)
        # output_dir = '/project/scratch/p200177/DE_371/victorsanchez/results/gan_training/exp_train_sequential_every_{config.timestep_period}h_10000_u_v_t2m_channel_multiplier=6/final_metrics/'
        if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        fig.savefig(output_dir+f'Diurne_Cycle_over_{nb_sample_total*16}_samples_{config.nb_timesteps}_nb_var_{len(config.var_names)}_{key}.png')

    else :
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
        ax.plot(range(config.nb_timesteps), value_at_given_pixel_per_variable_real_sample_t2m, linewidth=6, color='k', label='AROME')
        ax.set_ylabel('Temperature at 2m (K)', size=30)
        if not real_only:
            ax.plot(range(config.nb_timesteps), value_at_given_pixel_per_variable_generated_sample_t2m, linewidth=6, color='tab:orange', label='Generated')
        ax.legend(prop={'size':20})
        fig.suptitle(f'Diurnal Cycle on pixel {key}', size=30)
        # output_dir = '/project/scratch/p200177/DE_371/victorsanchez/results/gan_training/exp_train_sequential_every_{config.timestep_period}h_channel_multiplier_6/scores/plots/diurnal_cycle/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(output_dir+f'Diurne_Cycle_over_{nb_sample_total*16}_samples_{config.nb_timesteps}_nb_var_{len(config.var_names)}_{key}.png')

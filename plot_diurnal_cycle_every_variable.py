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
from tqdm import trange


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
    parser.add_argument('--training_dir', type=str, default='/project/scratch/p200177/DE_371/victorsanchez/results/gan_training/exp1bis_seq_GAN_exp_train_sequential_every_3h_channel_multiplier_2/scores/plots/diurnal_cycle/')
    parser.add_argument('--training_step', type=int, default=68000)
    # Model architecture hyper-parameters
    
    parser.add_argument('--model', type=str, default='stylegan2', \
                        choices=['stylegan2', 'stylegan2_fp16'])
    
    # choices of loss function and initialization
    parser.add_argument('--train_type', type=str, default='stylegan',\
                        choices=['stylegan', 'wave_d'])
    #architectural choices
    
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--g_channels', type=int, default=3)
    parser.add_argument('--d_channels', type=int, default=3)
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
    parser.add_argument('--config_dir', type=str, default="/project/scratch/p200177/DE_371/victorsanchez/results/gan_training/Set_UseNoiseFalse/", help="The config files absolute path")
    parser.add_argument('--dataset_handler_config', type=str, default="dataset_handler_config.yaml", help="The dataset_handler config file")
    parser.add_argument('--scheduler_config', type=str, default="scheduler_config.yaml", help="The scheduler config file")
    print("instantiating dataset")
    config = parser.parse_args()
    
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

    real_only = False

    if not real_only:
        print("instantiating generator")
        G = Generator(256, 512, n_mlp=8, nb_var=config.g_channels, channel_multiplier=config.channel_multiplier)
        
        ckpt = torch.load(config.training_dir+f'models/0{config.training_step}.pt', map_location='cpu')['g_ema']
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

    output_dir = config.training_dir+'scores/diurnal_cycle/'
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
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            fig.savefig(output_dir+f'Diurne_Cycle_over_{nb_sample_total*16}_samples_{config.nb_timesteps}_nb_var_{len(config.var_names)}_{key}.png')

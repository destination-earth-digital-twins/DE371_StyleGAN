#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:43:33 2022

@authors: gandonb, rabaultj, brochetc
"""
print('Importing Basic Package')
import os
import sys
import argparse
import yaml
from pathlib import Path
import gan.memutils.memory_consumption as memco
import gan.metrics4arome as METR
import gan.metrics4arome.spectrum_analysis as Spectral
import gan.metrics4arome.wasserstein_distances as WD
import metrics4arome.inception_metrics as inception
import gan.model.trainer_ddp as trainer
import gan.plot.plotting_functions as plf
import torch
from gan.distributed import is_main_gpu, synchronize
from gan.metrics4arome import sliced_wasserstein as SWD
from torch import distributed as dist
from torch.distributed import destroy_process_group, init_process_group

print(f'\n Torch Version : {torch.__version__}\n')

sys.stdout.reconfigure(line_buffering=True, write_through=True)
try:
    local_rank = int(os.environ["LOCAL_RANK"])
    
except KeyError:
    local_rank = 0

print(f'local_rank {local_rank}')


if torch.cuda.is_available():
    print('torch.cuda.is_available')
    torch.cuda.set_device(local_rank)
    init_process_group(
        'nccl' if dist.is_nccl_available() else 'gloo',
        rank=local_rank,
        world_size=torch.cuda.device_count()
    )


###############################################################################
############################# Parameters #########################
###############################################################################



def str2bool(v):
    return v.lower() in ('true')

def str2list(li):
    if type(li)==list:
        li2 = li
        return li2
    
    elif type(li)==str:
        li2=li[1:-1].split(',')
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))

def str2intlist(li):
    if type(li)==list:
        li2 = [int(p) for p in li]
        return li2
    
    elif type(li)==str:
        li2 = li[1:-1].split(',')
        li3 = [int(p) for p in li2]
        return li3

    else : 
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))

def str2inttuple(li):
    if type(li)==list:
        li2 =[int(p) for p in li]  
        return tuple(li2)
    
    elif type(li)==str:
        li2 = li[1:-1].split(',')
        li3 =[int(p) for p in li2]

        return tuple(li3)

    else : 
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))


def read_yamlconfig(config_file_abs_path):
    if Path(config_file_abs_path).is_file():  # exists
        print("config file found")
        with open(config_file_abs_path) as f: 
            optyaml = yaml.safe_load(f)  # load hyps
    else:
        raise NameError(f"config file {config_file_abs_path} not found")
    ensemble = optyaml["ensemble"]
    return  optyaml, ensemble, config_file_abs_path

def get_dirs(config_dir):


    """
    
    read config file for the experimentations
    
    """
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='main.yaml')
    args =   parser.parse_args()
    config_file_abs_path = f"{config_dir}{args.config_file}"
    return read_yamlconfig(config_file_abs_path)

def get_expe_parameters():

    parser = argparse.ArgumentParser()
    print('loading parser')
    # Paths
    parser.add_argument('--data_dir', type=str, default="/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/")
    #parser.add_argument('--data_dir', type=str, default="/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_large_lt_done/")
    parser.add_argument('--mean_file', type=str, default=None )
    parser.add_argument('--std_file', type=str, default=None )
    parser.add_argument('--max_file', type=str, default=None )
    parser.add_argument('--min_file', type=str, default=None )
    parser.add_argument('--id_file', type=str, default="Large_lt_train_labels_1.csv")
    parser.add_argument('--pretrained_model', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default='/project/home/p200177/DE_371/experiments_WP1/gan_training/exp_train_without_Noise_Injection/')

    # Model architecture hyper-parameters
    
    parser.add_argument('--model', type=str, default='stylegan2', \
                        choices=['stylegan2', 'stylegan2_fp16', 'stylegan2_3d'])
    
    parser.add_argument('--g_dim', type=str, default='2d', choices=['2d', '3d'])
    parser.add_argument('--d_dim', type=str, default='2d', choices=['2d', '3d'])
    
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
    parser.add_argument('--timestep_labelling', action='store_true')
    parser.add_argument('--variable_first', action='store_true')
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
    return parser

if __name__=="__main__" :
    ###############################################################################
    ############################# INITIALIZING EXPERIMENT #########################
    ###############################################################################
    print('INITIALIZING EXPERIMENT')
    # params = argparse.ArgumentParser().parse_args()
    config = get_expe_parameters().parse_args()
    # # Merging both configs
    # for param in params : 
    #     config[param] = params[param]

    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)
    if not os.path.exists(config.output_dir + "/log"):
        os.mkdir(config.output_dir + "/log")
    if not os.path.exists(config.output_dir + "/models"):
        os.mkdir(config.output_dir + "/models")
    if not os.path.exists(config.output_dir + "/samples"):
        os.mkdir(config.output_dir + "/samples")

    ########### write inversion parameters to file ############
    config_file = config.output_dir + "/training_params.yaml"
    print("writing params config file:", config_file)
    try:
        file=open(config_file,"w")
        yaml.dump(config.__dict__,file)
    except Exception as e:
         print("unable to write params config file")
         print(e)

    if config.g_dim == config.d_dim:
        if config.model=='stylegan2':
            import gan.model.stylegan2 as RN

        elif config.model=='stylegan2_fp16':
            import gan.model.stylegan2_fp16 as RN

        elif config.model=='stylegan2_3d':
            import gan.model.stylegan2_3d as RN

        else:
            raise ValueError('Model unknown')

    else :
        import gan.model.stylegan2 as RN_2D
        import gan.model.stylegan2_3d as RN_3D
        # import gan.model.stylegan2_fp16 as RN_fp16

    ###############################################################################
    ############################ BUILDING MODELS ##################################
    ###############################################################################

    load_optim = False

    if config.g_dim != config.d_dim :
        #TODO : Implement that on the trainer
        raise NotImplementedError
    
    try:

        if config.train_type=='stylegan':

            if config.g_dim == config.d_dim:
                model_names = RN.library[config.model]
                modelG_n, modelD_n = getattr(RN, model_names['G']), getattr(RN, model_names['D'])
            else :
                model_names_2d = RN_2D.library[config.model]
                model_names_3d = RN_3D.library[config.model]
                if config.g_dim == '2d':
                    modelG_n = getattr(RN_2D, model_names_2d['G'])
                    modelD_n = getattr(RN_3D, model_names_3d['D'])
                elif config.g_dim == '3d':
                    modelG_n = getattr(RN_3D, model_names_3d['G'])
                    modelD_n = getattr(RN_2D, model_names_2d['D'])
                else :
                    raise NotImplementedError
                
            if config.model in ['stylegan2','stylegan2_3d']:

                if config.g_dim == '2d' :
                    modelG = modelG_n(config.crop_size[0], config.latent_dim, config.n_mlp,
                                    channel_multiplier=config.channel_multiplier, 
                                    nb_var=config.g_channels,# if not config.mean_pert else len(config.var_names)*2,
                                    var_rr=('rr' in config.var_names),
                                    tanh_output=config.tanh_output,
                                    use_noise=config.use_noise)

                    modelG_ema = modelG_n(config.crop_size[0], config.latent_dim, config.n_mlp,
                                    channel_multiplier=config.channel_multiplier, 
                                    nb_var=config.g_channels,# if not config.mean_pert else len(config.var_names)*2,
                                    var_rr=('rr' in config.var_names),
                                    tanh_output=config.tanh_output,
                                    use_noise=config.use_noise)
                   
                elif config.g_dim == '3d':
                    if not config.variable_first:
                        modelG = modelG_n(config.crop_size[0], config.latent_dim, config.n_mlp,
                                                channel_multiplier=config.channel_multiplier, 
                                                nb_var=config.nb_timesteps, # Channel
                                                nb_frames=config.g_channels, # Depth
                                                var_rr=('rr' in config.var_names),
                                                tanh_output=config.tanh_output,
                                                use_noise=config.use_noise)
                            
                        modelG_ema = modelG_n(config.crop_size[0], config.latent_dim, config.n_mlp,
                                                channel_multiplier=config.channel_multiplier, 
                                                nb_var=config.nb_timesteps, # Channel
                                                nb_frames=config.g_channels, # Depth
                                                var_rr=('rr' in config.var_names),
                                                tanh_output=config.tanh_output,
                                                use_noise=config.use_noise)
                    else :
                        modelG = modelG_n(config.crop_size[0], config.latent_dim, config.n_mlp,
                                                channel_multiplier=config.channel_multiplier, 
                                                nb_var=config.g_channels, # Channel
                                                nb_frames=config.nb_timesteps, # Depth
                                                var_rr=('rr' in config.var_names),
                                                tanh_output=config.tanh_output,
                                                use_noise=config.use_noise)
                            
                        modelG_ema = modelG_n(config.crop_size[0], config.latent_dim, config.n_mlp,
                                                channel_multiplier=config.channel_multiplier, 
                                                nb_var=config.g_channels, # Channel
                                                nb_frames=config.nb_timesteps, # Depth
                                                var_rr=('rr' in config.var_names),
                                                tanh_output=config.tanh_output,
                                                use_noise=config.use_noise)
                else :
                    raise NotImplementedError

                if config.d_dim == '2d' or (config.d_dim == '3d' and config.variable_first):
                    print(f'Init Discriminator with : {config.d_channels} channels')
                    modelD = modelD_n(config.crop_size[0],
                                channel_multiplier=config.channel_multiplier, 
                                    nb_var=config.d_channels)# if not config.mean_pert else len(config.var_names)*2)
                
                elif config.d_dim == '3d' and not config.variable_first:
                    print(f'Init Discriminator with : {config.nb_timesteps} channels')
                    modelD = modelD_n(config.crop_size[0],
                                channel_multiplier=config.channel_multiplier, 
                                    nb_var=config.nb_timesteps)
                else :

                    raise NotImplementedError
   
            elif config.model=='stylegan2_fp16':

                modelG = modelG_n(config.crop_size[0], config.latent_dim, config.n_mlp,
                                channel_multiplier=config.channel_multiplier,
                                num_fp16_res=config.fp16_resolution)

                modelD = modelD_n(config.crop_size[0],
                                channel_multiplier=config.channel_multiplier,
                                num_fp16_res=config.fp16_resolution)

                modelG_ema = modelG_n(config.crop_size[0], config.latent_dim, config.n_mlp,
                                    channel_multiplier=config.channel_multiplier,
                                    num_fp16_res=config.fp16_resolution)

        elif config.train_type=='wave_d':

            import gan.model.swagan as RN1 

            model_names = RN.library[config.model]

            model_names_sw = RN1.library['swagan']

            modelG_n = getattr(RN, model_names['G'])
            modelD_n = getattr(RN1, model_names_sw['D'])


            modelG = modelG_n(config.crop_size[0], config.latent_dim, config.n_mlp,
                                    channel_multiplier=config.channel_multiplier, 
                                    nb_var=len(config.var_names),
                                    var_rr=('rr' in config.var_names),
                                    tanh_output=config.tanh_output,# if not config.mean_pert else len(config.var_names)*2,
                                use_noise=config.use_noise)

            modelD = modelD_n(config.crop_size[0],
                                channel_multiplier=config.channel_multiplier,
                                nb_var=len(config.var_names),)

            modelG_ema = modelG_n(config.crop_size[0], config.latent_dim, config.n_mlp,
                                    channel_multiplier=config.channel_multiplier, 
                                    var_rr=('rr' in config.var_names),
                                    tanh_output=config.tanh_output,
                                    nb_var=len(config.var_names),# if not config.mean_pert else len(config.var_names)*2,
                                    use_noise=config.use_noise)

        else:

            modelG = modelG_n(config.latent_dim, config.g_channels)
            modelD = modelD_n(config.d_channels)

    except KeyError: # back to "default names", error-prone is not wished for!

        modelG = RN.ResNet_G(config.latent_dim, config.g_output_dim, config.g_channels)

        modelD = RN.ResNet_D(config.d_input_dim, config.d_channels)

    if config.pretrained_model>=0:

        i = config.pretrained_model
        print(i, config.output_dir + f'/models/{str(i).zfill(6)}.pt')
        ckpt = torch.load(config.output_dir + f'/models/{str(i).zfill(6)}.pt', map_location='cpu')
        ## BAZ
        ckpt["g"] = {key.replace("module.", ""): value for key, value in ckpt["g"].items()}
        ckpt["d"] = {key.replace("module.", ""): value for key, value in ckpt["d"].items()}

        modelG.load_state_dict(ckpt["g"])
        modelD.load_state_dict(ckpt["d"])

        if config.train_type == 'stylegan' or "wave_d":
            ckpt["g_ema"] = {key.replace("module.", ""): value for key, value in ckpt["g_ema"].items()}
            modelG_ema.load_state_dict(ckpt["g_ema"])

    else:

        ckpt = None

        modelG_ema.eval()

        trainer.accumulate(modelG_ema, modelG, 0)
    print('\n synchronizing')
    synchronize()
    print('\n synchronizing done')
    ###############################################################################
    ######################### Defining metrics #############################
    ###############################################################################

    # names used in test_metrics should belong to the metrics namespace --> on-the-fly definition of metrics
    
    setattr(METR, "fid", 
                            METR.metric2D('Fréchet Inception Distance  ',\
                                inception.FIDclass(inception.inceptionPath).FID,\
                                ['FID']))

    sliced_wd = SWD.SWD_API2(numpy=False, ch_per_ch=False)

    setattr(METR, "SWD_metric_torch", 
                            METR.metric2D('Sliced Wasserstein Distance  ',\
                                sliced_wd.End2End,\
                                [str(var_name) for var_name in config.var_names], 
                                names=sliced_wd.get_metric_names(),))
                                #mean_pert=config.mean_pert))

    setattr(METR, "spectral_dist_torch_"+"_".join(str(var_name) for var_name in config.var_names), 
                            METR.metric2D('Power Spectral Density RMSE', 
                                Spectral.PSD_compare_torch, 
                                [str(var_name) for var_name in config.var_names], 
                                names = [f'PSD{str(var)}' for var in config.var_names],))
                                #mean_pert=config.mean_pert))

    setattr(METR, "W1_center", 
                            METR.metric2D('Mean Wasserstein distance on center crop  ', 
                                WD.W1_center, 
                                [str(var_name) for var_name in config.var_names], 
                                names = ['W1_Center'],))
                                #mean_pert=config.mean_pert))

    setattr(METR, "W1_Random", 
                            METR.metric2D('Mean Wasserstein distance on random selection  ', 
                                WD.W1_random, 
                                [str(var_name) for var_name in config.var_names], 
                                names = ['W1_random'],))
                                #mean_pert=config.mean_pert))


    test_metr = ["W1_Random", "SWD_metric_torch"] # if not config.mean_pert else ["W1_Random"] # SWD won't work with mean_pert
    #if not config.mean_pert:
    test_metr = test_metr + ["spectral_dist_torch_"+"_".join(str(var_name) for var_name in config.var_names)] # same (or at least need some work)

    ###############################################################################
    ######################### LOADING models and Data #############################
    ###############################################################################

    print('\n creating trainer', flush=True)
    TRAINER = trainer.Trainer(config, criterion="W1_center", test_metrics=test_metr)

    modelG, modelD, modelG_ema, mem_g, mem_d, mem_opt, mem_cuda = TRAINER.instantiate(modelG, modelD, load_optim=ckpt, modelG_ema=modelG_ema)

    # memco.log_mem_consumption(modelG, modelD, config, mem_g, mem_d, mem_opt, mem_cuda)



    ###############################################################################
    ################################## TRAINING ###################################
    ##########################   (and online testing)  ############################
    ###############################################################################

    TRAINER.fit_(modelG, modelD, modelG_ema=modelG_ema)

    ###############################################################################
    ############################## Light POST-PROCESSING ##########################
    ############################ (of training output data) ########################

    if is_main_gpu():
        plf.plot_metrics_from_csv(config.output_dir + '/log/', 'metrics.csv')

    synchronize()

    destroy_process_group()
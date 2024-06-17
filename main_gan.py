#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:43:33 2022

@authors: gandonb, rabaultj, brochetc
"""

import os
import sys

import gan.memutils.memory_consumption as memco
import gan.metrics4arome as METR
import gan.metrics4arome.spectrum_analysis as Spectral
import gan.metrics4arome.wasserstein_distances as WD
import gan.model.trainer_ddp as trainer
import gan.plot.plotting_functions as plf
import torch
from gan.distributed import get_rank, get_world_size, is_main_gpu, synchronize
from expe_init import get_expe_parameters
from gan.metrics4arome import sliced_wasserstein as SWD
from torch import distributed as dist
from torch.distributed import destroy_process_group, init_process_group

print(f'\n{torch.__version__}\n')

sys.stdout.reconfigure(line_buffering=True, write_through=True)
try:
    local_rank = int(os.environ["LOCAL_RANK"])
except KeyError:
    local_rank = 0
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    init_process_group(
        'nccl' if dist.is_nccl_available() else 'gloo',
        rank=local_rank,
        world_size=torch.cuda.device_count())


###############################################################################
############################# INITIALIZING EXPERIMENT #########################
###############################################################################

config = get_expe_parameters().parse_args()
if not os.path.exists(config.output_dir):
    os.mkdir(config.output_dir)
if not os.path.exists(config.output_dir + "/log"):
    os.mkdir(config.output_dir + "/log")
if not os.path.exists(config.output_dir + "/models"):
    os.mkdir(config.output_dir + "/models")
if not os.path.exists(config.output_dir + "/samples"):
    os.mkdir(config.output_dir + "/samples")

if config.model=='stylegan2':
    import gan.model.stylegan2 as RN

elif config.model=='stylegan2_fp16':
    import gan.model.stylegan2_fp16 as RN

else:
    raise ValueError('Model unknown')

###############################################################################
############################ BUILDING MODELS ##################################
###############################################################################

load_optim = False

try:

    if config.train_type=='stylegan':

        model_names = RN.library[config.model]

        modelG_n, modelD_n = getattr(RN, model_names['G']), getattr(RN, model_names['D'])

        if config.model=='stylegan2':

            modelG = modelG_n(config.crop_size[0], config.latent_dim, config.n_mlp,
                                channel_multiplier=config.channel_multiplier, 
                                nb_var=len(config.var_names),# if not config.mean_pert else len(config.var_names)*2,
                                var_rr=('rr' in config.var_names),
                                tanh_output=config.tanh_output,
                                use_noise=config.use_noise)

            modelD = modelD_n(config.crop_size[0],
                              channel_multiplier=config.channel_multiplier, 
                                nb_var=len(config.var_names),)# if not config.mean_pert else len(config.var_names)*2)

            modelG_ema = modelG_n(config.crop_size[0], config.latent_dim, config.n_mlp,
                                channel_multiplier=config.channel_multiplier, 
                                nb_var=len(config.var_names),# if not config.mean_pert else len(config.var_names)*2,
                                var_rr=('rr' in config.var_names),
                                tanh_output=config.tanh_output,
                                use_noise=config.use_noise)
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

synchronize()

###############################################################################
######################### Defining metrics #############################
###############################################################################

# names used in test_metrics should belong to the metrics namespace --> on-the-fly definition of metrics

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

print('creating trainer', flush=True)
TRAINER = trainer.Trainer(config,criterion="W1_center",\
                        test_metrics=test_metr)

print('instantiating', flush=True)
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

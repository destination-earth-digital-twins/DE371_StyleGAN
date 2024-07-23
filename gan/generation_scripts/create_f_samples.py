import os
import sys

HOME_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(HOME_DIR)

from multiprocessing import Pool
from time import perf_counter

import metrics4arome as METR
import model.stylegan2 as RN
import numpy as np
import torch
import yaml

"""
Created on Tue Apr 11 14:24:10 2023
@author: poulainl
""" 
print("Running")

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

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake samples creator")

    parser.add_argument("set", type=int, help='set')
    parser.add_argument("config_file", type=str, help="Name of the config_file")
    parser.add_argument("num_samples", type=int, help="Number of samples generated for each checkpoint")
    parser.add_argument("--name", type=str, default='', help="suffix for sample folder")

    parser.add_argument("--ckpt", type=str2intlist, default=[0, ], help="--ckpt='[0, 50000, 10000]' wil start from 0, end at 50000 and evaluate every 10000 checkpoint between")
    parser.add_argument("--nb_batch", type=int, default=128, help="Number of _F_sample.npy file for each checkpoint")
    parser.add_argument("--ch_mul", type=int, default=2, help="Channels multiplier")
    parser.add_argument("--rgb", action="store_true", help="save the different outputs of the ToRGB layers")
    parser.add_argument("--no_mean", action="store_true", help="dataset with mean image taken off")
    parser.add_argument("--mean_pert", action="store_true", help="dataset split between mean and pert")
    parser.add_argument("--rgb_levels", type=str2intlist, default=[0,1,2,3], help="which keys of rgb dict to save. Refer to the stylegan.py code for indices reference")
    parser.add_argument("--instance", type=int, default=1, help='instance')

    args = parser.parse_args()

    chm = args.ch_mul

    with open(os.path.join(HOME_DIR, "configs", f"Set_{args.set}", args.config_file), 'r') as main_config_yaml:
        config = yaml.safe_load(main_config_yaml)

    lat_dim = config["ensemble"]["--latent_dim"][0]
    bs = config["ensemble"]["--batch_size"][0]
    lr_G = config["ensemble"]["--lr_G"][0]
    lr_D = config["ensemble"]["--lr_D"][0]
    size = str2inttuple(config["ensemble"]["--crop_indexes"][0])[1] - str2inttuple(config["ensemble"]["--crop_indexes"][0])[0]
    use_noise = config["ensemble"]["--use_noise"][0]
    var_names = str2list(config["ensemble"]["--var_names"][0])
    tanh_output = config["ensemble"]["--tanh_output"][0]


    assert not (args.no_mean and args.mean_pert), "Can't have 0_mean and mean_pert activated at the same time"
    no_mean = "_no_mean" if args.no_mean else ""
    mean_pert = "_mean_pert" if args.mean_pert else ""


    output_dir = os.path.join(config["output_dir"], f"Set_{args.set}", f"stylegan2_stylegan_dom_{size}_lat-dim_{lat_dim}_bs_{bs}_{lr_D}_{lr_G}_ch-mul_{chm}_vars_{'_'.join(str(var_name) for var_name in var_names)}_noise_{use_noise}/Instance_{args.instance}")
    sample_folder_name = f"samples{f'_{args.name}' if args.name != '' else ''}"
    os.makedirs(os.path.join(output_dir, sample_folder_name), exist_ok=True)
    rgb_keys = {0:'prev_rgb',1:'prev_rgb_upsampled',2:'input_conved', 3:'current_rgb_out'}
        
    for ckpt in range(args.ckpt[0], args.ckpt[1]+1, args.ckpt[2]):

        ckpt_path = os.path.join(output_dir, "models")
        ckpt_name = f"{str(ckpt).zfill(6)}.pt"
        model_names =  RN.library['stylegan2']

        device  = torch.device('cuda')
        num_samples = args.num_samples

        modelG_n = getattr(RN, model_names['G'])
        modelG_ema = modelG_n(size=size, style_dim=lat_dim, n_mlp=8, channel_multiplier=chm, nb_var=len(var_names)*2 if args.mean_pert else len(var_names), var_rr=('rr' in var_names), tanh_output=tanh_output, use_noise=use_noise)
        modelG_ema = modelG_ema.to(device)

        ckpt_dic = torch.load(os.path.join(ckpt_path, ckpt_name), map_location=device)
        ckpt_dic["g_ema"] = {key.replace("module.", ""): value for key, value in ckpt_dic["g_ema"].items()}
        modelG_ema.load_state_dict(ckpt_dic["g_ema"])
        modelG_ema.eval()
        nb_batch = args.nb_batch
        nb_batch = nb_batch*2 if args.rgb else nb_batch

        t_s = perf_counter()
        for j in range(nb_batch):
            if (j+1) % 16 == 0:
                print(f"batch {j+1}/{nb_batch}")
            z = torch.empty(num_samples//nb_batch, lat_dim).normal_().to(device)
            with torch.no_grad():
                fake_samples, _, rgb = modelG_ema([z], return_rgb=True)
                if not args.rgb:
                    np.save(os.path.join(output_dir, sample_folder_name, f"_Fsample_{int(ckpt)}_{j}.npy"), fake_samples.cpu().numpy())
                if args.rgb:
                    if not os.path.isdir(os.path.join(output_dir, "toRGB_outs")):
                        os.mkdir(os.path.join(output_dir, "toRGB_outs"))
                    for i in args.rgb_levels:
                        rgb_key = rgb_keys[i]
                        savename = os.path.join("toRGB_outs", f"RGBS_level_{rgb_key}_lat_{lat_dim}_bs_{bs}_chm_{chm}_lr_{lr_G}_ckpt_{ckpt}_{j}.npy")
                        np.save(os.path.join(output_dir, savename), rgb[rgb_key])
                        np.save(os.path.join(output_dir, "toRGB_outs", f"_Fsample_{int(ckpt)}_{j}.npy"), fake_samples.cpu().numpy())
        print(f"{num_samples} images produced in {perf_counter()-t_s}s")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import os
import argparse
from gan.model.stylegan2 import Generator
from collections import OrderedDict


if __name__=="__main__" :
        
    #%%
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_dir', type=str, default='/project/home/p200177/DE_371/resources/models/trained_generator/000024.pt')
    parser.add_argument('--output_dir', type=str, default='/project/scratch/p200177/DE_371/victorsanchez/results/pca/w_samples')

    args = parser.parse_args()
    output_dir = args.output_dir
    ckpt_dir = args.ckpt_dir
    print("ckpt_dir:", ckpt_dir)
    print("output_dir:", output_dir)


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir + "/w"):
        os.makedirs(output_dir+"/w")
    if not os.path.exists(output_dir + "/z"):
        os.makedirs(output_dir+"/z")
    if not os.path.exists(output_dir + "/x"):
        os.makedirs(output_dir+"/x")

    #%%
    print("instantiating generator")
    G = Generator(256, 512, n_mlp=8, nb_var=3)

    ckpt = torch.load(ckpt_dir, map_location='cpu')['g_ema']
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

    #%%
    print("random sampling of latent codes from generator")
    n = 1000
    for i in range(n):
        if (i+1)%100==0:
            print(i+1)
        if not (os.path.isfile(output_dir + '/w/_w_{}.npy'.format(i)) and os.path.isfile(output_dir + '/x/_x_{}.npy'.format(i))):

            z = torch.empty(14, 512).normal_().to(device)

            with torch.no_grad():
                x, w, _ = G([z], return_latents=True)
                x = x.cpu().numpy()
                w = w.cpu().numpy()

            z = z.cpu().detach().numpy()
            np.save(output_dir + '/w/_w_{}.npy'.format(i), w)
        #    np.save(output_dir + '/z/_z_{}.npy'.format(i), z)
            np.save(output_dir + '/x/_x_{}.npy'.format(i), x)


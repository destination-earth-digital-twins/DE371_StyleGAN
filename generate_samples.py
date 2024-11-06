import os
import torch
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import time  # Pour mesurer le temps
import torch
import argparse
from torchvision.utils import save_image
from gan.model.stylegan2 import Generator
import json
from time import perf_counter
import yaml
import pandas as pd
from datetime import date, timedelta, datetime
import perturbation.utils as utils
import pickle
import matplotlib.pyplot as plt


torch.manual_seed(42) #reproducibility of runs




# Appel de la fonction avec les paramètres appropriés
# This file generate n samples from the GAN 

if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################
    # Checkpoint directory - PATH to generator's weight
    parser.add_argument('--ckpt_dir', type = str, 
                        default ='/project/scratch/p200177/DE_371/angeliquebonamy/gan_training/exp_train_ep_with_Noise_Injection/models/138000.pt')
    parser.add_argument('--output_dir',type = str, 
                        default ='/project/scratch/p200177/DE_371/angeliquebonamy/gan_training/exp_train_ep_with_Noise_Injection/ggenerated_samples')
   
    parser.add_argument('--device', type=str, default='cuda:0')

    ############################ INVERSION PARAMETERS #################    

    parser.add_argument("--var_indices", type=utils.str2intlist, default=[0,1,2,3])
    parser.add_argument("--Shape", type=tuple, default=(4,256,256), help='size of the samples')
    
    parser.add_argument("--seed", type=int, default=42)
    
    params = parser.parse_args()
    # Initialisation du périphérique
    device = params.device if torch.cuda.is_available() else 'cpu'

    # Initialisation du générateur G
    G = Generator(params.Shape[1], 512, n_mlp=8, nb_var=params.Shape[0])
    print('JE SUIS PARAMS', params.Shape[1], params.Shape[0], params.ckpt_dir)

    # Chargement du checkpoint
    ckpt = torch.load(params.ckpt_dir, map_location='cpu')['g_ema']
    if 'module' in list(ckpt.items())[0][0]:  # Adaptation pour les modules empaquetés
        ckpt_adapt = OrderedDict()
        for k in ckpt.keys():
            k0 = k[7:]
            ckpt_adapt[k0] = ckpt[k]
        G.load_state_dict(ckpt_adapt)
    else:
        G.load_state_dict(ckpt)

    G.eval()
    G = G.to(device)

    # Création des dossiers de sortie
    if not os.path.exists(os.path.join(params.output_dir)):
        os.makedirs(os.path.join(params.output_dir))

    # Calcul de la moyenne du style latent, ou chargement s'il existe déjà
    latent_path = f'{os.path.join(params.output_dir)}/latent_mean.npy'
    if not os.path.exists(latent_path):
        latent_z = torch.empty(10000, 512).normal_().to(device)
        with torch.no_grad():
            w = G.style(latent_z)
        latent_mean = w.mean(dim=0).detach().cpu()
        np.save(latent_path, latent_mean.numpy())
    else:
        lm = np.load(latent_path).astype(np.float32)
        latent_mean = torch.tensor(lm, dtype=torch.float32)

    # Nombre total d'échantillons
    total_samples = 16384

    # Initialisation de la barre de progression
    with tqdm(total=total_samples, desc="Génération des samples", unit="sample") as pbar:
        start_time = time.time()
        for sample_idx in range(total_samples):
            # Génère un vecteur latent unique
            latent_z = torch.empty(1, 512).normal_().to(device)
            with torch.no_grad():
                style = G.style(latent_z)
                generated_image = G([style])

            # Sauvegarde du sample unique
            sample_path = os.path.join(params.output_dir, f'sample_{sample_idx}.npy')
            np.save(sample_path, generated_image[0].squeeze(0).detach().cpu().numpy())  # Sauvegarde chaque sample séparément

            # Mise à jour de la barre de progression
            pbar.update(1)
            elapsed_time = time.time() - start_time
            pbar.set_postfix({"Temps écoulé": f"{elapsed_time:.2f} sec"})

        print("Génération complète de 16,384 samples terminée.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ##########################################################
# This script is to do a PCA on the different latent space W
# ##########################################################

# ## OBS! Before running this script, make sure you have run mkl_w_sample.py with "ckpt_dir" set accordingly

import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from gan.model.stylegan2 import Generator
from collections import OrderedDict
import torch 
import torch.nn.functional as F

def PCA(X , num_components=None):

    if len(X.shape) != 2 :
        raise NotImplementedError
    
    if num_components is None:
        num_components=np.min(X.shape)
     
    # Mean of data
    X_meaned = X - np.mean(X , axis = 0)
     
    # Covariance Matrix
    cov_mat = np.cov(X_meaned , rowvar = False) 
    # intercor_mat = np.correlate(X_meaned.flatten(),X_meaned.flatten())
     
    # Eigen Values and Eigen Vectors
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    # Selecting only the principal eigenvectors (that have the highest eigenvalues)
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    # Projecting our initial data along the principal components axis
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced, sorted_eigenvalue, sorted_eigenvectors




# Do not hesitate to adapt the code depending on what you want to do 
#Argument parser 
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir',        type=str, default='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt')
parser.add_argument('--output_dir',      type=str, default='/project/scratch/p200177/DE_371/victorsanchez/results/pca/pca_test')
parser.add_argument('--w_inversion_dir1', type=str, default='/project/scratch/p200177/DE_371/inversion_experiments/exp34/inversion')
parser.add_argument('--w_samples_dir',   type=str, default='/project/scratch/p200177/DE_371/victorsanchez/results/pca/w_samples') # samples generated with mkl_w_sample.py
parser.add_argument('--inversion_step',  type=str, default="2000")
parser.add_argument('--case',           type=str, default="2021-07-01", help="specific inversion case to consider") # %Y-%m-%d_lt
args = parser.parse_args()


## Loading files
## From gan
files_w = glob.glob(f"{args.w_samples_dir}/w/_w*.npy")

## 16 members inverted with Perceptual Loss and MSE
w_inv1 = np.load(f"{args.w_inversion_dir1}/w_{args.case}_3_{args.inversion_step}.npy") # (16,14,512)
w_inv2 = np.load(f"{args.w_inversion_dir1}/w_{args.case}_12_{args.inversion_step}.npy") # (16,14,512)

import torch
#%%
print("instantiating generator")
G = Generator(256, 512, n_mlp=8, nb_var=3)

ckpt_dir = args.ckpt_dir
ckpt = torch.load(ckpt_dir, map_location='cpu')['g_ema']
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if 'module' in list(ckpt.items())[0][0]: # juglling with Pytorch versioning and different module packaging
     ckpt_adapt = OrderedDict()
     for k in ckpt.keys():
         k0 = k[7:]
         ckpt_adapt[k0] = ckpt[k]
     G.load_state_dict(ckpt_adapt)
else:
     G.load_state_dict(ckpt)

G.eval()
G = G.cuda()


## load random w samples generated from mapping network of gan
w_samples = []
print("loading w samples")
for f in files_w:
    w_sample=np.load(f)
    if w_sample.ndim<3: # (B, 512)
        w_samples.append(w_sample[0,:])
    else: # (B, 14, 512)
        w_samples.append(w_sample[0,0,:])

## PCA samples generated from GAN
num_components = 3
print(f"pca on w_samples on {num_components} principal axis")
w_samples_reduced, w_samples_sorted_eigenvalue, w_samples_sorted_eigenvectors = PCA(X=np.array(w_samples), num_components=num_components)
prop_var_w_samples = w_samples_sorted_eigenvalue / np.sum(w_samples_sorted_eigenvalue)

eigenvector_subset = w_samples_sorted_eigenvectors[:,0:num_components]
w_inv1_projected = np.array([np.dot(eigenvector_subset.transpose() , (w_inv1[i]-np.mean(w_inv1[i], axis=0)).transpose() ).transpose() for i in range(16)])
w_inv2_projected = np.array([np.dot(eigenvector_subset.transpose() , (w_inv2[i]-np.mean(w_inv2[i], axis=0)).transpose() ).transpose() for i in range(16)])

img_from_inv1 = G([torch.from_numpy(w_inv1).cuda()], input_is_latent=True)[0].cpu().detach().numpy()
distance_latent = [F.mse_loss(torch.from_numpy(w_inv1_projected[0]), torch.from_numpy(w_inv1_projected[i])) for i in range(0,16)]
distance_real = [F.mse_loss(torch.from_numpy(img_from_inv1[0]), torch.from_numpy(img_from_inv1[i])) for i in range(0,16)]

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot()
for i in range(len(distance_latent)):
    ax.scatter(distance_latent[i], distance_real[i])
    ax.text(distance_latent[i], distance_real[i],  '%s' % (str(i)), size=20, zorder=1,  color='red') 
ax.set_xlabel('latent distance')
ax.set_ylabel('real distance')
fig.suptitle('Latent distance over Real distance',size = 25)
figname = f"{args.output_dir}/latent_distance_{args.case}_step_{args.inversion_step}.png"
fig.savefig(figname)


import torch
#%%
print("instantiating generator")
G = Generator(256, 512, n_mlp=8, nb_var=3)

ckpt_dir = args.ckpt_dir
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

member_0 = w_inv1[0]
member_1 = w_inv1[1]
index = 7
w_hybrid = np.concatenate((member_0[:index], member_1[index:]))
print('shape w_hybrid', np.shape(w_hybrid))
img_from_w_hybrid = G([torch.from_numpy(w_hybrid).unsqueeze(0).cuda()], input_is_latent=True)[0].cpu().detach().numpy()[0]
img_from_w_member_0 = G([torch.from_numpy(member_0).unsqueeze(0).cuda()], input_is_latent=True)[0].cpu().detach().numpy()[0]
img_from_w_member_1 = G([torch.from_numpy(member_1).unsqueeze(0).cuda()], input_is_latent=True)[0].cpu().detach().numpy()[0]
print(np.shape(img_from_w_hybrid))
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(24,24))
var = ['u', 'v', 't2m']
for var_id in range(len(img_from_w_hybrid)):
    ax[0][var_id].imshow(img_from_w_member_0[var_id], origin="lower")
    ax[0][var_id].set_title(var[var_id] + ' member0')
    ax[1][var_id].imshow(img_from_w_hybrid[var_id], origin="lower")
    ax[1][var_id].set_title(var[var_id] + ' member hybrid')
    ax[2][var_id].imshow(img_from_w_member_1[var_id], origin="lower")
    ax[2][var_id].set_title(var[var_id] + ' member1')


fig.suptitle('Latent distance over Real distance',size = 25)
figname = f"{args.output_dir}/hybrid_member_{args.case}_step_{args.inversion_step}.png"
fig.savefig(figname)
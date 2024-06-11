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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch 

# Do not hesitate to adapt the code depending on what you want to do 
#Argument parser 
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir',        type=str, default='/scratch/mrmn/brochetc/GAN_2D/Exp_StyleGAN_final/Set_1/stylegan2_stylegan_dom_256_lat-dim_512_bs_4_0.002_0.002_ch-mul_2_vars_u_v_t2m_noise_True/Instance_14/models/000024.pt')
parser.add_argument('--output_dir',      type=str, default='/scratch/mrmn/sanchezv/project/results/Ens_Perceptual_Random_VGG_Loss')
parser.add_argument('--w_inversion_dir1', type=str, default='/scratch/mrmn/sanchezv/project/results/Ens_Perceptual_Random_VGG_Loss/Inversion_Perceptual_Random_VGG_Loss')
parser.add_argument('--w_inversion_dir2', type=str, default='/scratch/mrmn/sanchezv/project/results/Ens_MSE_Loss/Inversion_MSE_Loss')
parser.add_argument('--w_samples_dir',   type=str, default='/scratch/mrmn/sanchezv/project/results/Ens_Perceptual_Random_VGG_Loss/w_samples') # samples generated with mkl_w_sample.py
parser.add_argument('--inversion_step',  type=str, default="2000")
parser.add_argument('--case',           type=str, default="2021-07-05_6", help="specific inversion case to consider") # %Y-%m-%d_lt
parser.add_argument('--arome_875_w_samples_file',  type=str, default='/scratch/mrmn/sanchezv/project/results/Latent_Sapce_Analysis/data/Inversion_GE/w_ge_3_875.npy') # samples generated with mkl_w_sample.py
args = parser.parse_args()

## Loading files
## From gan
files_w = glob.glob(f"{args.w_samples_dir}/w/_w*.npy")
files_x = glob.glob(f"{args.w_samples_dir}/x/_x*.npy")

## 16 members inverted with Perceptual Loss and MSE
w_inv1 = np.load(f"{args.w_inversion_dir1}/w_{args.case}_{args.inversion_step}.npy")

## 16 members inverted with MSE
# w_inv2 = np.load(f"{args.w_inversion_dir2}/w_{args.case}_{args.inversion_step}.npy")

## 875 AROME members inverted with MSE
w_875_arome = np.load(args.arome_875_w_samples_file)

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
print("pca on w samples")
pca = PCA(n_components=512)
pca_w_samples = pca.fit(np.array(w_samples))

eigenvalues_w_samples = pca_w_samples.explained_variance_
prop_var_w_samples = eigenvalues_w_samples / np.sum(eigenvalues_w_samples)

print("loading w_inv1")
if w_inv1.ndim<3: # (B, 512)
    w_inv1 = w_inv1[:,:]
else: # (B, 14, 512)
    w_inv1 = w_inv1[:,0,:]

print("pca on w_inv1")
pca = PCA(n_components=16)
pca_w_inv1 = pca.fit(np.array(w_inv1))

eigenvalues_w_inv1 = pca_w_inv1.explained_variance_
prop_var_w_inv1 = eigenvalues_w_inv1 / np.sum(eigenvalues_w_inv1)


# print("loading w_inv2")
# if w_inv2.ndim<3:
    # # w_inv2 = w_inv2[:,:]
# else:
    # # w_inv2 = w_inv2[:,0,:]

# print("pca on w_inv2")
# pca = PCA(n_components=n_components)
# # pca_w_inv2 = pca.fit(np.array(w_inv2))

# # eigenvalues_w_inv2 = pca_w_inv2.explained_variance_
# # # prop_var_w_inv2 = eigenvalues_w_inv2 / np.sum(eigenvalues_w_inv2)

# print("loading w_inv2")
if w_875_arome.ndim<3:
    w_875_arome = w_875_arome[:,:]
else:
    w_875_arome = w_875_arome[:,0,:]

print("pca on w_875_arome")
pca = PCA(n_components=512)
pca_w_875_arome = pca.fit(np.array(w_875_arome))

eigenvalues_w_875_arome = pca_w_875_arome.explained_variance_
prop_var_w_875_arome = eigenvalues_w_875_arome / np.sum(eigenvalues_w_875_arome)

eigenvectors_w_875_arome = torch.from_numpy(pca_w_875_arome.components_)
w_875_arome_norm = torch.from_numpy(np.array(w_875_arome)-np.mean(np.array(w_875_arome), axis=0))
projection = (torch.einsum('ab, db-> da', eigenvectors_w_875_arome, w_875_arome_norm)/torch.sqrt(torch.from_numpy(eigenvalues_w_875_arome).view(1,512))).numpy()

mean_projection = projection.mean(axis=0)
std_projection = np.std(projection, axis=0, ddof=1)

plt.figure(figsize=(14,10))
plt.plot(np.arange(1, len(mean_projection)+1), mean_projection, marker='o', c='blue')
plt.fill_between(np.arange(1, len(mean_projection)+1), mean_projection-std_projection, mean_projection+std_projection, alpha=0.3, color='magenta')
# plt.plot(np.arange(1, len(prop_var_w_inv1)+1), prop_var_w_inv1, marker='o', c='red')
# # plt.plot(np.arange(1, len(prop_var_w_inv2)+1), prop_var_w_inv2, marker='o', c='green')
plt.xlabel('Principal Component',size = 20)
# plt.yscale('log')
plt.ylabel('Projection ',size = 20)
plt.title('Figure 0: Projection',size = 25)
plt.grid(True)

figname = f"{args.output_dir}/project_875_scree_plot_w_scatter_{args.case}_step_{args.inversion_step}.png"
plt.savefig(figname, dpi=100)

# plt.figure(figsize=(14,10))
# plt.plot(np.arange(1, len(prop_var_w_samples)+1), prop_var_w_samples, marker='o', c='blue')
# plt.plot(np.arange(1, len(prop_var_w_875_arome)+1), prop_var_w_875_arome, marker='o', c='magenta')
# # plt.plot(np.arange(1, len(prop_var_w_inv1)+1), prop_var_w_inv1, marker='o', c='red')
# # # plt.plot(np.arange(1, len(prop_var_w_inv2)+1), prop_var_w_inv2, marker='o', c='green')
# plt.xlabel('Principal Component',size = 20)
# plt.yscale('log')
# plt.ylabel('Proportion of Variance Explained',size = 20)
# plt.title('Figure 1: Scree Plot for Proportion of Variance Explained',size = 25)
# plt.grid(True)

# figname = f"{args.output_dir}/875_scree_plot_w_scatter_{args.case}_step_{args.inversion_step}.png"
# plt.savefig(figname, dpi=100)

# # kaiser rule
# plt.figure(figsize=(14,10))
# plt.plot(np.arange(1, len(eigenvalues_w_samples)+1), eigenvalues_w_samples, marker='o', c='blue')
# # plt.plot(np.arange(1, len(eigenvalues_w_inv1)+1), eigenvalues_w_inv1, marker='o',  c='red')
# plt.plot(np.arange(1, len(eigenvalues_w_875_arome)+1), eigenvalues_w_875_arome, marker='o',  c='magenta')
# # # plt.plot(np.arange(1, len(eigenvalues_w_inv2)+1), eigenvalues_w_inv2, marker='o', c='green')
# plt.xlabel('Principal Component',size = 20)
# plt.ylabel('Eigenvalue',size = 20)
# plt.yscale('log')
# plt.title('Figure 2: Scree Plot for Eigenvalues',size = 25)
# plt.axhline(y=1, color='r',linestyle='--')
# plt.grid(True)
# figname = f"{args.output_dir}/875_scree_plot_kaiser_w_scatter_{args.case}_step_{args.inversion_step}.png"
# plt.savefig(figname, dpi=100)

# plt.figure(figsize=(7,7))
# plt.scatter(pca_w_samples.components_[0],pca_w_samples.components_[1],cmap='prism', s=5, c='blue')
# plt.scatter(pca_w_875_arome.components_[0],pca_w_875_arome.components_[1],cmap='prism', s=5, c='magenta')
# # plt.scatter(pca_w_inv1.components_[0],pca_w_inv1.components_[1],cmap='prism', s=5, c='red')
# # # plt.scatter(pca_w_inv2.components_[0],pca_w_inv2.components_[1],cmap='prism', s=5, c='green')
# plt.xlabel('pc1')
# plt.ylabel('pc2')
# plt.title('Figure 3: PCA on first two components',size = 25)
# figname = f"{args.output_dir}/875_pca_w_scatter_{args.case}_step_{args.inversion_step}.png"
# plt.savefig(figname, dpi=100)

# fig = plt.figure(figsize=(7,7))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(pca_w_samples.components_[0],pca_w_samples.components_[1],pca_w_samples.components_[2],cmap='prism', c='blue')
# ax.scatter(pca_w_875_arome.components_[0],pca_w_875_arome.components_[1],pca_w_875_arome.components_[2],cmap='prism', c='magenta')
# # ax.scatter(pca_w_inv1.components_[0],pca_w_inv1.components_[1],pca_w_inv1.components_[2],cmap='prism', c='red')
# # # plt.scatter(pca_w_inv2.components_[0],pca_w_inv2.components_[1],cmap='prism', s=5, c='green')
# ax.set_xlabel('pc1')
# ax.set_ylabel('pc2')
# ax.set_zlabel('pc3')
# ax.set_title('Figure 4: PCA on first three components',size = 25)
# figname = f"{args.output_dir}/875_pca_3d_w_scatter_{args.case}_step_{args.inversion_step}.png"
# fig.savefig(figname, dpi=100)
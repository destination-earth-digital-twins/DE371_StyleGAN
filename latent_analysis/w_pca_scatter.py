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
parser.add_argument('--output_dir',      type=str, default='/project/scratch/p200177/DE_371/victorsanchez/results/pca/pca_classic')
parser.add_argument('--w_inversion_dir1', type=str, default='/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Validation_Set_For_ScaleTune/Inversion')
parser.add_argument('--w_inversion_dir2', type=str, default='') 
parser.add_argument('--w_samples_dir',   type=str, default='/project/scratch/p200177/DE_371/victorsanchez/results/pca/w_samples') # samples generated with mkl_w_sample.py
parser.add_argument('--inversion_step',  type=str, default="2000")
parser.add_argument('--case',           type=str, default="2020-07-01_3", help="specific inversion case to consider") # %Y-%m-%d_lt
parser.add_argument('--arome_875_w_samples_file',  type=str, default='/scratch/mrmn/sanchezv/project/results/Latent_Space_Analysis/data/Inversion_GE/w_ge_3_875.npy') # samples generated with mkl_w_sample.py
args = parser.parse_args()

## Loading files
## From gan
files_w = glob.glob(f"{args.w_samples_dir}/w/_w*.npy")
files_x = glob.glob(f"{args.w_samples_dir}/x/_x*.npy")

## 16 members inverted with Perceptual Loss and MSE
w_inv1 = np.load(f"{args.w_inversion_dir1}/w_{args.case}_{args.inversion_step}.npy")

## 16 members inverted with MSE
if args.w_inversion_dir2 :
    w_inv2 = np.load(f"{args.w_inversion_dir2}/w_{args.case}_1000.npy")
else :
    w_inv2=None



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

print("loading w_inv1")
if w_inv1.ndim<3: # (B, 512)
    w_inv1 = w_inv1[:,:]
else: # (B, 14, 512)
    w_inv1 = w_inv1[:,0,:]

if w_inv2 is not None:
    print("loading w_inv2")
    if w_inv2.ndim<3:
        w_inv2 = w_inv2[:,:]
    else:
        w_inv2 = w_inv2[:,0,:]

# print("loading w_875")
# if w_875_arome.ndim<3:
#     w_875_arome = w_875_arome[:,:]
# else:
#     w_875_arome = w_875_arome[:,0,:]

## 875 AROME members inverted with MSE
# w_875_arome = np.load(args.arome_875_w_samples_file)

eigenvector_subset = w_samples_sorted_eigenvectors[:,0:num_components]
w_inv1_projected = np.dot(eigenvector_subset.transpose() , (w_inv1-np.mean(w_inv1, axis=0)).transpose() ).transpose()

if w_inv2 is not None:
    w_inv2_projected = np.dot(eigenvector_subset.transpose() , (w_inv2-np.mean(w_inv2, axis=0)).transpose() ).transpose()

fig, ax = plt.subplots(figsize=(21,7),nrows=1,ncols=3)
ax[0].scatter(w_samples_reduced[:,0],w_samples_reduced[:,1],cmap='prism', c='blue', alpha=0.1)
ax[1].scatter(w_samples_reduced[:,0],w_samples_reduced[:,2],cmap='prism', c='blue', alpha=0.1)
ax[2].scatter(w_samples_reduced[:,1],w_samples_reduced[:,2],cmap='prism', c='blue', alpha=0.1)
for i in range(len(w_inv1_projected)):
    ax[0].scatter(w_inv1_projected[i,0], w_inv1_projected[i,1],cmap='prism', c='red')
    if w_inv2 is not None:
        ax[0].scatter(w_inv2_projected[i,0], w_inv2_projected[i,1],cmap='prism', c='green')
    # ax[0].text(w_inv1_projected[i,0], w_inv1_projected[i,1],  '%s' % (str(i)), size=20, zorder=1,  color='red') 
    # ax[0].text(w_inv2_projected[i,0],w_inv2_projected[i,1],  '%s' % (str(i)), size=20, zorder=1,  color='green')

    ax[1].scatter(w_inv1_projected[i,0], w_inv1_projected[i,2],cmap='prism', c='red')
    if w_inv2 is not None:
        ax[1].scatter(w_inv2_projected[i,0], w_inv2_projected[i,2],cmap='prism', c='green')
    # ax[1].text(w_inv1_projected[i,0], w_inv1_projected[i,2],  '%s' % (str(i)), size=20, zorder=1,  color='red') 
    # ax[1].text(w_inv2_projected[i,0],w_inv2_projected[i,2],  '%s' % (str(i)), size=20, zorder=1,  color='green')

    ax[2].scatter(w_inv1_projected[i,1], w_inv1_projected[i,2],cmap='prism', c='red')
    if w_inv2 is not None:
        ax[2].scatter(w_inv2_projected[i,1], w_inv2_projected[i,2],cmap='prism', c='green')
    # ax[2].text(w_inv1_projected[i,1], w_inv1_projected[i,2],  '%s' % (str(i)), size=20, zorder=1,  color='red') 
    # ax[2].text(w_inv2_projected[i,1],w_inv2_projected[i,2],  '%s' % (str(i)), size=20, zorder=1,  color='green')
ax[0].set_xlabel('pc1')
ax[0].set_ylabel('pc2')
# ax[0].set_label(['Perceptual+MSE', 'MSE'])
ax[1].set_xlabel('pc1')
ax[1].set_ylabel('pc3')
# ax[1].set_label(['Perceptual+MSE', 'MSE'])
ax[2].set_xlabel('pc2')
ax[2].set_ylabel('pc3')
# ax[2].set_label(['Perceptual+MSE', 'MSE'])
fig.suptitle('Figure 1: PCA on first three components',size = 25)
figname = f"{args.output_dir}/projected_pca_2d_w_scatter_{args.case}_step_{args.inversion_step}.png"
fig.savefig(figname, dpi=100)



# import torch
# #%%
# print("instantiating generator")
# G = Generator(256, 512, n_mlp=8, nb_var=3)

# ckpt_dir = args.ckpt_dir
# ckpt = torch.load(ckpt_dir, map_location='cpu')['g_ema']
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("using device:", device)

# if 'module' in list(ckpt.items())[0][0]: # juglling with Pytorch versioning and different module packaging
#      ckpt_adapt = OrderedDict()
#      for k in ckpt.keys():
#          k0 = k[7:]
#          ckpt_adapt[k0] = ckpt[k]
#      G.load_state_dict(ckpt_adapt)
# else:
#      G.load_state_dict(ckpt)

# G.eval()
# G = G.to(device)

# z = torch.empty(14, 512).normal_().to(device)
# img_from_inv1 = G([w_inv1], input_is_latent=True, noise=z)[0]
# img_from_inv2 = G([w_inv2], input_is_latent=True, noise=z)[0]

# variables = ['u', 'v', 't2m']
# for var in range(3):
#     fig = plt.figure(figsize=(16,16))
#     ax = fig.add_subplot(nrows=4,ncols=4)
#     member_id=0
#     for i in range(4):
#         for j in range(4):
#             vmin=np.min(img_from_inv1[var][i][j])
#             vmax=np.max(img_from_inv1[var][i][j])
#             ax[i][j].imshow(img_from_inv1[var][i][j], clim=(vmin, vmax), origin="lower")
#             ax[i][j].set_title(str(member_id))
#             member_id+=1
#         figname = f"{args.output_dir}/inv_sample_from_Percep_"+variables[var]+f"_{args.case}_step_{args.inversion_step}.png"
#         fig.savefig(figname, dpi=100)
#     plt.close()

# for var in range(3):
#     fig = plt.figure(figsize=(16,16))
#     ax = fig.add_subplot(nrows=4,ncols=4)
#     member_id=0
#     for i in range(4):
#         for j in range(4):
#             vmin=np.min(img_from_inv2[var][i][j])
#             vmax=np.max(img_from_inv2[var][i][j])
#             ax[i][j].imshow(img_from_inv2[var][i][j], clim=(vmin, vmax), origin="lower")
#             ax[i][j].set_title(str(member_id))
#             member_id+=1
#         figname = f"{args.output_dir}/inv_sample_from_MSE_"+variables[var]+f"_{args.case}_step_{args.inversion_step}.png"
#         fig.savefig(figname, dpi=100)
#     plt.close()

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(projection='3d')
ax.scatter(w_samples_reduced[:,0],w_samples_reduced[:,1],w_samples_reduced[:,2],cmap='prism', c='blue', alpha=0.1)
# ax.scatter(pca_w_875_arome.components_[0],pca_w_875_arome.components_[1],pca_w_875_arome.components_[2],cmap='prism', c='magenta')
for i in range(len(w_inv1_projected)):
    ax.scatter(w_inv1_projected[i,0], w_inv1_projected[i,1], w_inv1_projected[i,2], cmap='prism', c='red')
    if w_inv2 is not None :
        ax.scatter(w_inv2_projected[i,0], w_inv2_projected[i,1], w_inv2_projected[i,2], cmap='prism', c='green')
    # ax.text(w_inv1_projected[i,0], w_inv1_projected[i,1], w_inv1_projected[i,2], '%s' % (str(i)), size=20, zorder=1,  color='red') 
    # ax.text(w_inv2_projected[i,0], w_inv2_projected[i,1], w_inv2_projected[i,2], '%s' % (str(i)), size=20, zorder=1,  color='green')
ax.set_xlabel('pc1')
ax.set_ylabel('pc2')
ax.set_zlabel('pc3')
fig.suptitle('Figure 2: PCA on first three components 3d',size = 25)
figname = f"{args.output_dir}/projected_pca_3d_w_scatter_{args.case}_step_{args.inversion_step}.png"
fig.savefig(figname, dpi=100)

# # Visualize all components

num_components=512
w_samples_reduced, w_samples_sorted_eigenvalue, w_samples_sorted_eigenvectors = PCA(X=np.array(w_samples), num_components=num_components)
prop_var_w_samples = w_samples_sorted_eigenvalue / np.sum(w_samples_sorted_eigenvalue)

num_components=16
w_inv1_reduced, w_inv1_sorted_eigenvalue, w_inv1_sorted_eigenvectors = PCA(X=w_inv1, num_components=num_components)
prop_var_w_inv1 = w_inv1_sorted_eigenvalue / np.sum(w_inv1_sorted_eigenvalue)

fig = plt.figure(figsize=(16,7))
ax = fig.add_subplot()
ax.plot(np.arange(1, len(prop_var_w_samples)+1), prop_var_w_samples, marker='o', c='blue')
# ax.plot(np.arange(1, len(prop_var_w_875_arome)+1), prop_var_w_875_arome, marker='o', c='magenta')
ax.plot(np.arange(1, len(prop_var_w_inv1)+1), prop_var_w_inv1, marker='o', c='red')
# # ax.plot(np.arange(1, len(prop_var_w_inv2)+1), prop_var_w_inv2, marker='o', c='green')
ax.set_xlabel('Principal Component',size = 20)
ax.set_yscale('log')
ax.set_ylabel('Proportion of Variance Explained',size = 20)
fig.suptitle('Figure 3: Scree Plot for Proportion of Variance Explained',size = 25)
ax.grid(True)
figname = f"{args.output_dir}/scree_plot_w_scatter_{args.case}_step_{args.inversion_step}.png"
fig.savefig(figname, dpi=100)


fig = plt.figure(figsize=(16,7))
ax = fig.add_subplot()
ax.plot(np.arange(1, len(w_samples_sorted_eigenvalue)+1), w_samples_sorted_eigenvalue, marker='o', c='blue')
# ax.plot(np.arange(1, len(prop_var_w_875_arome)+1), prop_var_w_875_arome, marker='o', c='magenta')
ax.plot(np.arange(1, len(w_inv1_sorted_eigenvalue)+1), w_inv1_sorted_eigenvalue, marker='o', c='green')
# # ax.plot(np.arange(1, len(prop_var_w_inv2)+1), prop_var_w_inv2, marker='o', c='green')
ax.set_xlabel('Principal Component',size = 20)
ax.set_yscale('log')
ax.set_ylabel('Proportion of Variance Explained',size = 20)
fig.suptitle('Figure 4: Scree Plot for Eigenvalues',size = 25)
ax.axhline(y=1, color='r',linestyle='--')
ax.grid(True)
figname = f"{args.output_dir}/scree_plot_kaiser_w_scatter_{args.case}_step_{args.inversion_step}.png"
fig.savefig(figname, dpi=100)
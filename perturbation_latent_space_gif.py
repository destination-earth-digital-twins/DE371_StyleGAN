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

def projection(vector_to_project, eigenvector_subset):
    return np.dot(eigenvector_subset.transpose() , vector_to_project.transpose() ).transpose()

def deprojection(vector_to_deproject, eigenvector_subset):
    return np.dot(eigenvector_subset.transpose() , vector_to_deproject.transpose()).transpose()

# Do not hesitate to adapt the code depending on what you want to do 
#Argument parser 
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir',        type=str, default='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt')
parser.add_argument('--output_dir',      type=str, default='/project/scratch/p200177/DE_371/victorsanchez/results/pca/pca_test/')
parser.add_argument('--w_inversion_dir1', type=str, default='/project/scratch/p200177/DE_371/inversion_process_analysis/inversion/exp39/inversion')
parser.add_argument('--w_samples_dir',   type=str, default='/project/scratch/p200177/DE_371/victorsanchez/results/pca/w_samples') # samples generated with mkl_w_sample.py
parser.add_argument('--inversion_step',  type=str, default="2000")
parser.add_argument('--case',           type=str, default="2021-07-01", help="specific inversion case to consider") # %Y-%m-%d_lt
args = parser.parse_args()


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

## Loading files
## From gan
files_w = glob.glob(f"{args.w_samples_dir}/w/_w*.npy")

## 16 members inverted with Perceptual Loss and MSE
w_inv1 = np.load(f"{args.w_inversion_dir1}/w_{args.case}_3_{args.inversion_step}.npy") # (16,14,512)
w_inv2 = np.load(f"{args.w_inversion_dir1}/w_{args.case}_12_{args.inversion_step}.npy") # (16,14,512)


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
num_components = 2
print(f"pca on w_samples on {num_components} principal axis")
w_samples_reduced, w_samples_sorted_eigenvalue, w_samples_sorted_eigenvectors = PCA(X=np.array(w_samples), num_components=num_components)
prop_var_w_samples = w_samples_sorted_eigenvalue / np.sum(w_samples_sorted_eigenvalue)
diff_flag = False
member_id = 0
for first_pca_axis in range(0,5):
    for second_pca_axis_id in range(first_pca_axis, 5):
        eigenvector_subset = np.concatenate((np.expand_dims(w_samples_sorted_eigenvectors[:,first_pca_axis], axis=1),np.expand_dims(w_samples_sorted_eigenvectors[:,second_pca_axis_id], axis=1)), axis=1)
        w_member = (w_inv1[member_id]-np.mean(w_inv1[member_id], axis=0))
        img_from_w_member = G([torch.from_numpy(w_inv1[member_id]).unsqueeze(0).cuda()], input_is_latent=True)[0].cpu().detach().numpy()[0]
        variables = ['u', 'v', 't2m']
        style_vector = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        for var_id, var in enumerate(variables):
            print(var)
            fig, ax = plt.subplots(nrows=7, ncols=7, figsize=(56,56))
            for id_ax1, perturbation_intensity_pca1 in enumerate(np.linspace(-2,2,7)) :
                for id_ax2, perturbation_intensity_pca2 in enumerate(np.linspace(-2,2,7)) :
                    perturbation_on_pca1 = np.array(style_vector).reshape((14,1))*perturbation_intensity_pca1
                    perturbation_on_pca2 = np.array(style_vector).reshape((14,1))*perturbation_intensity_pca2
                    projected_perturbation = np.concatenate((perturbation_on_pca1, perturbation_on_pca2), axis=1)
                    perturbation = np.dot(projected_perturbation, eigenvector_subset.T)

                    latent_w = (w_inv1[member_id] + perturbation).astype(np.float32)
                    img_from_perturbated_w_member = G([torch.from_numpy(latent_w).unsqueeze(0).cuda()], input_is_latent=True)[0].cpu().detach().numpy()[0]
                    if diff_flag :
                        diff = img_from_perturbated_w_member[var_id] - img_from_w_member[var_id]
                    if not diff_flag and var=='t2m':
                        cmap = 'coolwarm'
                    elif not diff_flag :
                        cmap = 'viridis'
                    else :
                        cmap='RdYlGn'
                    if diff_flag :
                        ax[id_ax1][id_ax2].imshow(diff,  origin="lower", cmap=cmap, clim=(-0.5,0.5))
                    else :
                        ax[id_ax1][id_ax2].imshow(img_from_perturbated_w_member[var_id],  origin="lower", cmap=cmap)
                    ax[id_ax1][id_ax2].set_title('pert {} | var {}'.format((perturbation_intensity_pca1,perturbation_intensity_pca2), var), fontsize=20)
            if diff_flag :
                fig.suptitle(f"diff_perturbation of {var} on style : {style_vector}", fontsize=60)
                fig.savefig(args.output_dir+f'diff_perturbation_of_{var}_{style_vector}_axis_{(first_pca_axis,second_pca_axis_id)}.png')
            else :
                fig.suptitle(f"perturbation of {var} on style : {style_vector}", fontsize=60)
                fig.savefig(args.output_dir+f'perturbation_of_{var}_{style_vector}_pca_axis_{(first_pca_axis,second_pca_axis_id)}.png')
            plt.close() 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ##########################################################
# This script is to do a PCA on the different latent space W
# ##########################################################

# ## OBS! Before running this script, make sure you have run mkl_w_sample.py with "ckpt_dir" set accordingly
from test_plot_temporal_sample import create_frame
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
num_components = 1
print(f"pca on w_samples on {num_components} principal axis")
w_samples_reduced, w_samples_sorted_eigenvalue, w_samples_sorted_eigenvectors = PCA(X=np.array(w_samples), num_components=num_components)
prop_var_w_samples = w_samples_sorted_eigenvalue / np.sum(w_samples_sorted_eigenvalue)



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


style_id = 13
member_id = 0
step = 0
frames = list()

eigenvector_subset = w_samples_sorted_eigenvectors[:,0:num_components]
w_member = (w_inv1[member_id]-np.mean(w_inv1[member_id], axis=0))
projected_w_member = projection(w_member, eigenvector_subset)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(24,24))
perturbation = np.zeros_like(projected_w_member)
img_from_w_member = G([torch.from_numpy(w_inv1[member_id]).unsqueeze(0).cuda()], input_is_latent=True)[0].cpu().detach().numpy()[0]
var = ['u', 'v', 't2m']
for var_id in range(3):
    ax[1][var_id].imshow(img_from_w_member[var_id],  origin="lower")
    ax[1][var_id].set_title('original '+var[var_id])

for perturbation_intensity in np.arange(0,7,0.25) :
    print('pertubating with intensity : ', perturbation_intensity)
    perturbation[style_id] = perturbation_intensity
    projected_w_member_perturbated = projected_w_member + perturbation
    unprojected_w_member = (np.dot(eigenvector_subset, projected_w_member_perturbated.T).T+np.mean(w_inv1[member_id], axis=0)).astype(np.float32)
    img_from_perturbated_w_member = G([torch.from_numpy(unprojected_w_member).unsqueeze(0).cuda()], input_is_latent=True)[0].cpu().detach().numpy()[0]
    for var_id in range(3):
        ax[0][var_id].imshow(img_from_perturbated_w_member[var_id],  origin="lower")
        ax[0][var_id].set_title(f'{perturbation_intensity} : perturbated '+var[var_id])
    fig.suptitle(f"perturbation intensity : {perturbation_intensity} on style : {style_id}")
    step+=1
    frames.append(create_frame(fig))


frame_one = frames[0]
frame_one.save(
    '/project/scratch/p200177/DE_371/victorsanchez/results/temporal_gif/test_perturbation/' + f"plot_perturbation_style_{style_id}.gif",
    format="GIF",
    append_images=frames,
    save_all=True,
    duration=5*step,
    loop=0,
)
plt.close() 

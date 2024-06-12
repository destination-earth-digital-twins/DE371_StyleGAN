#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## OBS! Before running this script, make sure you have run mkl_w_sample.py with "ckpt_dir" set accordingly

import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--ckpt_dir',        type=str, default='/scratch/mrmn/brochetc/GAN_2D/Exp_StyleGAN_final/Set_1/stylegan2_stylegan_dom_256_lat-dim_512_bs_4_0.002_0.002_ch-mul_2_vars_u_v_t2m_noise_True/Instance_14/models/000024.pt')
parser.add_argument('--output_dir',      type=str, default='/scratch/mrmn/sanchezv/project/results/Ens_Perceptual_Random_VGG_Loss')
parser.add_argument('--w_inversion_dir1', type=str, default='/scratch/mrmn/sanchezv/project/results/Ens_Perceptual_Random_VGG_Loss/Inversion_Perceptual_Random_VGG_Loss')
parser.add_argument('--w_inversion_dir2', type=str, default='/scratch/mrmn/sanchezv/project/results/Ens_MSE_Loss/Inversion_MSE_Loss')
parser.add_argument('--w_samples_dir',   type=str, default='/scratch/mrmn/sanchezv/project/results/Ens_Perceptual_Random_VGG_Loss/w_samples') # samples generated with mkl_w_sample.py
parser.add_argument('--inversion_step',  type=str, default="2000")
parser.add_argument('--case1',           type=str, default="2021-07-05_6", help="specific inversion case to consider") # %Y-%m-%d_lt
parser.add_argument('--case2',           type=str, default='2021-07-05_6')

args = parser.parse_args()
if args.case2=="":
   args.case2 = args.case1

output_dir = args.output_dir
ckpt_dir = args.ckpt_dir
samples_dir = args.w_samples_dir
inv_dir1 = args.w_inversion_dir1
inv_dir2 = args.w_inversion_dir2
inv_step = args.inversion_step
case1 = args.case1
case2 = args.case2

print("ckpt_dir:   ", ckpt_dir)
print("output_dir: ", output_dir)
print("case1:      ", case1)
print("case2:      ", case2)

files_w = glob.glob(f"{samples_dir}/w/_w*.npy")
files_x = glob.glob(f"{samples_dir}/x/_x*.npy")

## load random w samples from gan
W = []
print("loading w samples")
for f in files_w:
    w = np.load(f)
    if w.ndim<3: # (B, 512)
        W += [w[0,:]]
    else: # (B, 14, 512)
        W += [w[0,0,:]]

## load w samples from inversion
print("loading w samples from inversion")
w_inv1 = np.load(f"{inv_dir1}/w_{case1}_{inv_step}.npy")
w_inv2 = np.load(f"{inv_dir2}/w_{case2}_{inv_step}.npy")

if w_inv1.ndim<3: # (B, 512)
    w_inv1 = w_inv1[:,:]
else: # (B, 14, 512)
    w_inv1 = w_inv1[:,0,:]

if w_inv2.ndim<3:
    w_inv2 = w_inv2[:,:]
else:
    w_inv2 = w_inv2[:,0,:]

## count how many inversion components which are outside min,max range of w from gan
mem_idx = 0

for w_inv in [w_inv1, w_inv2]:
   n_inv_out = 0
   k_inv_out = []

   for kk in range(512):
       w_k = []
       for Wi in W:
           w_k += [Wi[kk]]

       w_inv_k = w_inv[mem_idx,kk]
       if w_inv_k < np.min(w_k) or w_inv_k > np.max(w_k):
           n_inv_out += 1
           k_inv_out += [kk]

   print("n_inv_out: ", n_inv_out)

## scatterplots for some components

k_plot = [] # using k's found for member mem_idx set above so not quite optimal since they are not the same for all members

nrows = 3
ncols = 3
n_plots = nrows*ncols
for i in range(2*n_plots):
    if i<len(k_inv_out):
        k_plot.append(k_inv_out[i])
    else:
        k_plot.append(np.random.randint(512))
k1_plot = k_plot[:int(len(k_plot)/2)]
k2_plot = k_plot[int(len(k_plot)/2):]


fig, axs = plt.subplots(nrows, ncols)
for i, ax in enumerate(axs.ravel()):
    k1 = k1_plot[i]
    k2 = k2_plot[i]

    w_k1 = [Wi[k1] for Wi in W]
    w_k2 = [Wi[k2] for Wi in W]

    ax.scatter(w_k1/np.linalg.norm(w_k1), w_k2/np.linalg.norm(w_k2), color="blue")

    for member in range(w_inv[:,:].shape[0]):
        w_inv1_k1, w_inv1_k2 = w_inv1[member,k1]/np.linalg.norm(w_inv1[member]), w_inv1[member, k2]/np.linalg.norm(w_inv1[member])
        w_inv2_k1, w_inv2_k2 = w_inv2[member,k1]/np.linalg.norm(w_inv2[member]), w_inv2[member, k2]/np.linalg.norm(w_inv2[member])
        ax.scatter(w_inv1_k1, w_inv1_k2, color="green")
        ax.scatter(w_inv2_k1, w_inv2_k2, color="red")

fig.set_figheight(12)
fig.set_figwidth(12)
figname = f"{output_dir}/w_scatter_{case1}_{case2}_step_{inv_step}_norm.png"
fig.savefig(figname, dpi=100)

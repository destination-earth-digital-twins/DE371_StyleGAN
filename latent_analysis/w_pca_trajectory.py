

import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

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


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir',        type=str, default='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt')
parser.add_argument('--output_dir',      type=str, default='/project/scratch/p200177/DE_371/victorsanchez/results/pca/pca_over_time')
parser.add_argument('--w_inversion_dir', type=str, default='/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3/Inversion_Perceptual_Random_VGG_Loss_sol3')
parser.add_argument('--w_samples_dir',   type=str, default='/project/scratch/p200177/DE_371/victorsanchez/results/pca/w_samples') # samples generated with mkl_w_sample.py
parser.add_argument('--inversion_step',  type=str, default="2000")
parser.add_argument('--date',           type=str, default="2021-07-05", help="specific inversion case to consider") # %Y-%m-%d_lt
parser.add_argument('--leadtimes',           type=str, default=[3,6,9,12,15,18,21,24,27,30,33,36,39,42]) # 
args = parser.parse_args()


## From gan
files_w = glob.glob(f"{args.w_samples_dir}/w/_w*.npy")
files_x = glob.glob(f"{args.w_samples_dir}/x/_x*.npy")

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
print(prop_var_w_samples[:num_components])
inv_samples = []
for leadtime in args.leadtimes :
    w_inv = np.load(f"{args.w_inversion_dir}/w_{args.date}_{leadtime}_{args.inversion_step}.npy")
    if w_inv.ndim<3: # (B, 512)
        w_inv = w_inv[:,:]
    else: # (B, 14, 512)
        w_inv = w_inv[:,0,:]
    inv_samples.append(w_inv)

inv_samples_proj = []
for inv_sample in inv_samples :
    inv_samples_proj.append(np.dot((inv_sample-np.mean(inv_sample, axis=0)),eigenvector_subset))
inv_samples_proj=np.array(inv_samples_proj)
for member_id in range(16):
    trajectory_of_member = inv_samples_proj[:,member_id,:]


    cols = np.linspace(0,len(trajectory_of_member),len(trajectory_of_member))
    fig, ax = plt.subplots(figsize=(21,7),nrows=1,ncols=3)
    ax[0].scatter(w_samples_reduced[:,0],w_samples_reduced[:,1],cmap='prism', c='blue', alpha=0.1)
    ax[1].scatter(w_samples_reduced[:,0],w_samples_reduced[:,2],cmap='prism', c='blue', alpha=0.1)
    ax[2].scatter(w_samples_reduced[:,1],w_samples_reduced[:,2],cmap='prism', c='blue', alpha=0.1)
    ax[0].scatter(trajectory_of_member[:,0],trajectory_of_member[:,1],cmap='prism', c=cols)
    ax[1].scatter(trajectory_of_member[:,0],trajectory_of_member[:,2],cmap='prism', c=cols)
    ax[2].scatter(trajectory_of_member[:,1],trajectory_of_member[:,2],cmap='prism', c=cols)
    # 0 and 1
    points = np.array([trajectory_of_member[:,0], trajectory_of_member[:,1]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='gist_rainbow')
    lc.set_array(cols)
    lc.set_linewidth(4)
    line = ax[0].add_collection(lc)

    # 0 and 2
    points = np.array([trajectory_of_member[:,0], trajectory_of_member[:,2]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='gist_rainbow')
    lc.set_array(cols)
    lc.set_linewidth(4)
    line = ax[1].add_collection(lc)

    # 1 and 2
    points = np.array([trajectory_of_member[:,1], trajectory_of_member[:,2]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='gist_rainbow')
    lc.set_array(cols)
    lc.set_linewidth(4)
    line = ax[2].add_collection(lc)
    ax[0].set_xlabel('pc1')
    ax[0].set_ylabel('pc2')
    ax[1].set_xlabel('pc1')
    ax[1].set_ylabel('pc3')
    ax[2].set_xlabel('pc2')
    ax[2].set_ylabel('pc3')
    fig.colorbar(line, ax=ax)
    figname = f"{args.output_dir}/projected_pca_3d_w_scatter_{args.date}_step_{args.inversion_step}_member_{member_id}.png"
    fig.suptitle(f"Trajectory of each leadtime sample of {args.date} projected over PCA on latent space of GAN")
    fig.savefig(figname, dpi=100)

# fig = plt.figure(figsize=(10,7))
# ax = fig.add_subplot(projection='3d')
# # ax.scatter(w_samples_reduced[:,0],w_samples_reduced[:,1],w_samples_reduced[:,2],cmap='prism', c='blue', alpha=0.1)
# im = ax.scatter(inv_samples_proj[:,0,0], inv_samples_proj[:,0,1], inv_samples_proj[:,0,2], linewidths=2, c=np.arange(len(inv_samples_proj[:,0,0])), cmap='viridis')
# ax.plot(inv_samples_proj[:,0,0], inv_samples_proj[:,0,1], inv_samples_proj[:,0,2], c='red')
# ax.set_xlabel('pc1')
# ax.set_ylabel('pc2')
# ax.set_zlabel('pc3')
# fig.colorbar(im)
# fig.tight_layout()
# fig.suptitle('PCA on first three components 3d',size = 25)
# figname = f"{args.output_dir}/projected_pca_3d_w_scatter_{args.date}_step_{args.inversion_step}.png"
# fig.savefig(figname, dpi=100)

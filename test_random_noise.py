import numpy as np
from inversion.optimization_based.inversion import latent_noise
from gan.model.stylegan2 import Generator
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
import pickle


G = Generator(256, 512,n_mlp=8,nb_var=3)
ckpt = torch.load('/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt', map_location='cpu')['g_ema']
# ckpt = torch.load('/project/scratch/p200177/DE_371/victorsanchez/results/gan_training/exp1/models/100000.pt', map_location='cpu')['g_ema']
pack_sample=np.load('/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise/Pack_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise/Rsemble_2021-07-07_3.npy')
inv_sample=np.load('/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise/Inversion_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise/invertFsemble_2021-07-07_3_2000.npy')



if 'module' in list(ckpt.items())[0][0]: #juglling with Pytorch versioning and different module packaging
    ckpt_adapt = OrderedDict()
    for k in ckpt.keys():
        k0 = k[7:]
        ckpt_adapt[k0] = ckpt[k]
    G.load_state_dict(ckpt_adapt)
else:
    G.load_state_dict(ckpt)

G.eval()
G = G.cuda()
output_dir = '/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise/Inversion_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise'

with torch.no_grad():
    noise_sample = torch.randn(pack_sample.shape[0], 512, device='cuda') # torch.Size([B,512]) (z)
    latent_out = G.style(noise_sample) # torch.Size([B,512]) (w)
    latent_mean = latent_out.mean(0) # mkl: this is weird. latent mean is passed as an input, but we are not using it ?
    latent_std = ((latent_out - latent_mean).pow(2).sum() / pack_sample.shape[0]) ** 0.5
with open('/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise/Inversion_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise/noise_2021-07-07_3_2000.p' , 'rb') as f:
    noises = pickle.load(f)
noise_vector=[]
for key in noises.keys():
    noise_vector.append(torch.from_numpy(np.array(noises[key])).cuda())

latent_in = np.load('/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise/Inversion_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise/w_2021-07-07_3_2000.npy')
latent_in=torch.from_numpy(latent_in).cuda()

Gen_fixed_noise = G([latent_in], input_is_latent=True, noise=noise_vector)[0].detach().cpu().numpy()
Gen_random_noise = G([latent_in], input_is_latent=True, noise=None)[0].detach().cpu().numpy()


fig = plt.figure(figsize=(15,15))
#### u
vmin = np.min([np.min(pack_sample[0,0,:,:])])
vmax = np.min([np.max(pack_sample[0,0,:,:])])

ax = fig.add_subplot(331)
ax.set_title("u real")
im = ax.imshow(pack_sample[0,0,:,:], clim=(vmin, vmax), origin="lower")
fig.colorbar(im, shrink=0.5)

ax = fig.add_subplot(334)
im = ax.imshow(Gen_fixed_noise[0,0,:,:], clim=(vmin, vmax), origin="lower")
ax.set_title("u fixed noise optim on random noise")
fig.colorbar(im, shrink=0.5)

ax = fig.add_subplot(337)
ax.set_title("u random noise optim on random noise")
im = ax.imshow(Gen_random_noise[0,0,:,:], clim=(vmin, vmax), origin="lower")
fig.colorbar(im, shrink=0.5)


#### v
vmin = np.min([np.min(pack_sample[0,1,:,:])])
vmax = np.min([np.max(pack_sample[0,1,:,:])])

ax = fig.add_subplot(332)
ax.set_title("v real")
im = ax.imshow(pack_sample[0,1,:,:], clim=(vmin, vmax), origin="lower")
fig.colorbar(im, shrink=0.5)

ax = fig.add_subplot(335)
im = ax.imshow(Gen_fixed_noise[0,1,:,:], clim=(vmin, vmax), origin="lower")
ax.set_title("v fixed noise optim on random noise")
fig.colorbar(im, shrink=0.5)

ax = fig.add_subplot(338)
ax.set_title("v random noise optim on random noise")
im = ax.imshow(Gen_random_noise[0,1,:,:], clim=(vmin, vmax), origin="lower")
fig.colorbar(im, shrink=0.5)


#### t2m
vmin = np.min([np.min(pack_sample[0,2,:,:])])
vmax = np.min([np.max(pack_sample[0,2,:,:])])

ax = fig.add_subplot(333)
im = ax.imshow(pack_sample[0,2,:,:], clim=(vmin, vmax), origin="lower")
ax.set_title("t2m real")
fig.colorbar(im, shrink=0.5)

ax = fig.add_subplot(336)
im = ax.imshow(Gen_fixed_noise[0,2,:,:], clim=(vmin, vmax), origin="lower")
ax.set_title("t2m fixed noise optim on random noise")
fig.colorbar(im, shrink=0.5)

ax = fig.add_subplot(339)
ax.set_title("t2m random optim on random noise")
im = ax.imshow(Gen_random_noise[0,2,:,:], clim=(vmin, vmax), origin="lower")
fig.colorbar(im, shrink=0.5)

fig.suptitle('Comparison of inverting samples from random and fixed noise')
fig.tight_layout()
fig.savefig('/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise/results_rd_not_fixed_noise_.png', dpi=100)


###########################"
# 
# 
G = Generator(256, 512,n_mlp=8,nb_var=3)
# ckpt = torch.load('/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt', map_location='cpu')['g_ema']
ckpt = torch.load('/project/scratch/p200177/DE_371/victorsanchez/results/gan_training/exp1/models/100000.pt', map_location='cpu')['g_ema']
pack_sample=np.load('/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise_no_noise_inj/Pack_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise_no_noise_inj/Rsemble_2021-07-07_3.npy')
inv_sample=np.load('/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise_no_noise_inj/Inversion_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise_no_noise_inj/invertFsemble_2021-07-07_3_2000.npy')



if 'module' in list(ckpt.items())[0][0]: #juglling with Pytorch versioning and different module packaging
    ckpt_adapt = OrderedDict()
    for k in ckpt.keys():
        k0 = k[7:]
        ckpt_adapt[k0] = ckpt[k]
    G.load_state_dict(ckpt_adapt)
else:
    G.load_state_dict(ckpt)

G.eval()
G = G.cuda()
output_dir = '/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise_no_noise_inj/Inversion_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise_no_noise_inj'

with torch.no_grad():
    noise_sample = torch.randn(pack_sample.shape[0], 512, device='cuda') # torch.Size([B,512]) (z)
    latent_out = G.style(noise_sample) # torch.Size([B,512]) (w)
    latent_mean = latent_out.mean(0) # mkl: this is weird. latent mean is passed as an input, but we are not using it ?
    latent_std = ((latent_out - latent_mean).pow(2).sum() / pack_sample.shape[0]) ** 0.5
with open('/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise_no_noise_inj/Inversion_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise_no_noise_inj/noise_2021-07-07_3_2000.p' , 'rb') as f:
    noises = pickle.load(f)
noise_vector=[]
for key in noises.keys():
    noise_vector.append(torch.from_numpy(np.array(noises[key])).cuda())

latent_in = np.load('/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise_no_noise_inj/Inversion_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise_no_noise_inj/w_2021-07-07_3_2000.npy')
latent_in=torch.from_numpy(latent_in).cuda()

Gen_fixed_noise = G([latent_in], input_is_latent=True, noise=noise_vector)[0].detach().cpu().numpy()
Gen_random_noise = G([latent_in], input_is_latent=True, noise=None)[0].detach().cpu().numpy()


fig = plt.figure(figsize=(15,15))
#### u
vmin = np.min([np.min(pack_sample[0,0,:,:])])
vmax = np.min([np.max(pack_sample[0,0,:,:])])

ax = fig.add_subplot(331)
ax.set_title("u real")
im = ax.imshow(pack_sample[0,0,:,:], clim=(vmin, vmax), origin="lower")
fig.colorbar(im, shrink=0.5)

ax = fig.add_subplot(334)
im = ax.imshow(Gen_fixed_noise[0,0,:,:], clim=(vmin, vmax), origin="lower")
ax.set_title("u fixed noise optim on random noise")
fig.colorbar(im, shrink=0.5)

ax = fig.add_subplot(337)
ax.set_title("u random noise optim on random noise")
im = ax.imshow(Gen_random_noise[0,0,:,:], clim=(vmin, vmax), origin="lower")
fig.colorbar(im, shrink=0.5)


#### v
vmin = np.min([np.min(pack_sample[0,1,:,:])])
vmax = np.min([np.max(pack_sample[0,1,:,:])])

ax = fig.add_subplot(332)
ax.set_title("v real")
im = ax.imshow(pack_sample[0,1,:,:], clim=(vmin, vmax), origin="lower")
fig.colorbar(im, shrink=0.5)

ax = fig.add_subplot(335)
im = ax.imshow(Gen_fixed_noise[0,1,:,:], clim=(vmin, vmax), origin="lower")
ax.set_title("v fixed noise optim on random noise")
fig.colorbar(im, shrink=0.5)

ax = fig.add_subplot(338)
ax.set_title("v random noise optim on random noise")
im = ax.imshow(Gen_random_noise[0,1,:,:], clim=(vmin, vmax), origin="lower")
fig.colorbar(im, shrink=0.5)


#### t2m
vmin = np.min([np.min(pack_sample[0,2,:,:])])
vmax = np.min([np.max(pack_sample[0,2,:,:])])

ax = fig.add_subplot(333)
im = ax.imshow(pack_sample[0,2,:,:], clim=(vmin, vmax), origin="lower")
ax.set_title("t2m real")
fig.colorbar(im, shrink=0.5)

ax = fig.add_subplot(336)
im = ax.imshow(Gen_fixed_noise[0,2,:,:], clim=(vmin, vmax), origin="lower")
ax.set_title("t2m fixed noise optim on random noise")
fig.colorbar(im, shrink=0.5)

ax = fig.add_subplot(339)
ax.set_title("t2m random optim on random noise")
im = ax.imshow(Gen_random_noise[0,2,:,:], clim=(vmin, vmax), origin="lower")
fig.colorbar(im, shrink=0.5)

fig.suptitle('Comparison of inverting samples from random and fixed noise')
fig.tight_layout()
fig.savefig('/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise_no_noise_inj/results_rd_fixed_noise_inversion_without_noise_inj.png', dpi=100)

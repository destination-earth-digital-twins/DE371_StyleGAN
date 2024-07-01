import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import perturbation.pca_stylegan as pca
import perturbation.smpca as smpca
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict
from gan.model.stylegan2 import Generator
from random import shuffle, uniform
import matplotlib.pyplot as plt
from itertools import product
from argparse import ArgumentParser
import os
#import spectral_loss_filtered as spec

#from hyperparams.util import str2intlist, load_all_lt, select_random_dates, load_whole_model, list_all_obs
# Tunning alpha (interp) and beta (scale) terms
from glob import glob

def add_noise(scale,sig,device):
   noise = torch.empty(scale.shape).normal_()
   return scale + sig * noise.to(device)

def sigma(t,epoch):
    return (0.2 /(1 + epoch)) * (1 - t**2)

def learning_rate(t, lr0):
    if t<0.25:
        return t * 0.01 + lr0
    elif t>0.5:
        return 0.01 * (1.0 - t)
    return 0.01

def convert_uvt2fft(batch_gen, batch_y):
    new_batch_gen = torch.cat((torch.sqrt(batch_gen[:,0:1,:,:]**2 + batch_gen[:,1:2,:,:]**2), batch_gen[:,2:,:,:]),dim=1)
    new_batch_y = torch.cat((torch.sqrt(batch_y[:,0:1,:,:]**2 + batch_y[:,1:2,:,:]**2),batch_y[:,2:,:,:]),dim=1)

    return new_batch_gen, new_batch_y

parser = ArgumentParser()

parser.add_argument("--n_epochs",type=int,default=20)
parser.add_argument("--n_samples",type=int,default=16)
parser.add_argument("--inflate_random",action="store_true")
parser.add_argument("--lr0",type=float, default=0.001)
parser.add_argument("--scale_rule",type=str,default='sigmoid')
parser.add_argument("--pca_cut",type=int,default=10)
parser.add_argument("--inflate",type=float, default=1.0)
parser.add_argument("--start",type=str, default="ones")
parser.add_argument("--lambda_bias",type=float, default=1.0)
parser.add_argument("--lambda_spectrum",type=float, default=0.0)
parser.add_argument("--lambda_spread",type=float, default=1.0)
parser.add_argument("--convert_ff_t",action="store_true")
parser.add_argument("--invert_step",type=int, default=1000)

########################### Directories ###########################
parser.add_argument("--fake_data_dir", type=str, 
                    default='/scratch/mrmn/brochetc/GAN_2D/Exp_StyleGAN_final/Inversion_Val/')
parser.add_argument("--real_data_dir", type=str, 
                    default='/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_large_lt_done2/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
parser.add_argument("--ensemble_data_dir", type=str, 
                    default='/scratch/mrmn/brochetc/GAN_2D/Exp_StyleGAN_final/Pack_Val/')
parser.add_argument("--ckpt_dir", type=str, 
                    default='/scratch/mrmn/brochetc/GAN_2D/Exp_StyleGAN_final/Set_1/stylegan2_stylegan_dom_256_lat-dim_512_bs_4_0.002_0.002_ch-mul_2_vars_u_v_t2m_noise_True/Instance_14/models/000024.pt')
parser.add_argument("--eigendir", type=str, 
                    default='/scratch/mrmn/brochetc/GAN_2D/Exp_StyleGAN_final/Eigenvalues/')
parser.add_argument("--output_dir", type=str, 
                    default='/scratch/mrmn/brochetc/GAN_2D/Exp_StyleGAN_final/ScaleTune/')

args = parser.parse_args()


output_dir = f"{args.output_dir}interp_scale_pca_{args.pca_cut}_{args.inflate_random}_{args.inflate}_bias_{args.start}_{args.lambda_bias}_spread_{args.lambda_spread}_ff_{args.convert_ff_t}_{args.invert_step}/"
os.makedirs(output_dir, exist_ok=True)
instances = len(glob(output_dir + "Instance_*/"))
print("instances already existing", instances)
os.makedirs(output_dir + f"Instance_{instances+1}/",exist_ok=True)
output_dir = output_dir + f"Instance_{instances+1}/"
df = pd.read_csv(args.real_data_dir + 'Large_lt_val_labels.csv')
df_date = df.copy()

liste_dates = df_date['Date'].unique().tolist()
print(liste_dates)
leadtimes = [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45]

ensemble_dataset = list(product(liste_dates,leadtimes))
print(len(ensemble_dataset))
print(ensemble_dataset[0])


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#specLoss = spec.SpectralLossFiltered(256,device)

Whitening = torch.load(args.eigendir + 'Whitening.pt').to(device)
Coloring = torch.load(args.eigendir + 'Coloring.pt').to(device)
w0 = torch.load(args.eigendir + 'latent_mean.pt').to(device)

print('loading G')

G = Generator(256, 512,n_mlp=8,nb_var=3)
#print('###########################################"##################################################################################################################')
ckpt = torch.load(args.ckpt_dir, map_location='cpu')['g_ema']
if 'module' in list(ckpt.items())[0][0]: #juglling with Pytorch versioning and different module packaging
    ckpt_adapt = OrderedDict()
    for k in ckpt.keys():
        k0 = k[7:]
        ckpt_adapt[k0] = ckpt[k]
    G.load_state_dict(ckpt_adapt)
else:
    G.load_state_dict(ckpt)
G.eval()
G = G.to(device)

print('G loaded')
if args.start=='ones':
    scale = torch.ones((14,),dtype=torch.float32, requires_grad=True,device=device)
    interp = 0.5 * torch.ones((14,),dtype=torch.float32,device=device)
elif args.start=='zeros':
    scale = torch.zeros((14,),dtype=torch.float32, requires_grad=True,device=device)
    interp = -1.0 * torch.ones((14,),dtype=torch.float32,device=device)
else:
    raise RuntimeError("Start unspecified")

interp = interp.requires_grad_()
optimizer = optim.Adam([interp, scale], lr = args.lr0)
track = [[],[],[],[],[]]
for epoch in range(args.n_epochs):
    shuffle(ensemble_dataset)
    print("#"*80)
    print(f"Epoch {epoch}")
    print("#"*80)
    pbar = tqdm(len(ensemble_dataset))
    for idx, (date,lt) in enumerate(ensemble_dataset):
        batch_w = torch.tensor(np.load(args.fake_data_dir + f"w_{date[:10]}_{lt}_{args.invert_step}.npy").astype(np.float32)).to(device)
        batch_y = torch.tensor(np.load(args.ensemble_data_dir + f"Rsemble_{date[:10]}_{lt}.npy").astype(np.float32)).to(device)
        
        t =  idx / len(ensemble_dataset) 
        optim.lr = learning_rate(t, args.lr0)
        scale_noise = add_noise(scale,sigma(t,epoch),device)
        interp_noise = add_noise(interp,sigma(t,epoch),device)

        try:
            Cov = torch.load(args.fake_data_dir + f'Cov_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')
            w_avg = torch.load(args.fake_data_dir + f'w_avg_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')

            if w_avg.shape!=(args.pca_cut,512):
                Cov, w_avg = pca.computeCovarianceW(batch_w[:,:args.pca_cut],cut=args.n_samples-1)
                torch.save(Cov, args.fake_data_dir + f'Cov_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')
                torch.save(w_avg, args.fake_data_dir + f'w_avg_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')

        except FileNotFoundError:
            Cov, w_avg  = pca.computeCovarianceW(batch_w[:,:args.pca_cut],cut=args.n_samples-1)
            torch.save(Cov, args.fake_data_dir + f'Cov_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')
            torch.save(w_avg, args.fake_data_dir + f'w_avg_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')
        
        try :
            assert w_avg.shape==(args.pca_cut,512)
        except AssertionError:
            print(date, lt, batch_w.shape, w_avg.shape)
            raise AssertionError("Uncorrect shape")

        gen = smpca.fast_style_mixing(interp_noise, scale_noise, batch_w, Cov,w_avg,w0,args.n_samples,G,Whitening,device=device,scale_rule=args.scale_rule) 
        if args.convert_ff_t:
            gen, batch_y = convert_uvt2fft(gen, batch_y)
        mean_loss = F.l1_loss(gen.mean(dim=0), batch_y.mean(dim=0))
        inflation = args.inflate if not args.inflate_random else (1.0 + uniform(0,args.inflate))
        std_loss = F.l1_loss(torch.std(gen,dim=0, unbiased=True), inflation * torch.std(batch_y,dim=0, unbiased=True))

        if args.lambda_spectrum>0.0:
            #spl = specLoss(batch_y,gen)
            loss = args.lambda_bias * mean_loss + args.lambda_spread * std_loss# \
            #            + args.lambda_spectrum * spl
        
        else:
            #with torch.no_grad():
            #    spl = specLoss(batch_y,gen)
            loss = args.lambda_bias * mean_loss + args.lambda_spread * std_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        emaloss = 0.9 * emaloss + 0.1 * loss.item() if idx>0 else loss.item()
        ema_spec = 0# 0.9 * ema_spec + 0.1 * spl.item() if idx>0 else spl.item()

        if args.scale_rule=='sigmoid':
            ema_scale = 0.1 * F.sigmoid(scale.detach().cpu()).numpy() + 0.9 * ema_scale if idx>0 else F.sigmoid(scale.detach().cpu()).numpy()
        else:
            ema_scale = 0.1 * scale.detach().cpu().numpy() + 0.9 * ema_scale if idx>0 else scale.detach().cpu().numpy()
        ema_interp = 0.1 * F.sigmoid(interp).detach().cpu().numpy() + 0.9 * ema_interp if idx>0 else F.sigmoid(interp).detach().cpu().numpy()

        with torch.no_grad():
            emabias = mean_loss.item() * 0.1 + 0.9 * emabias if idx>0 else mean_loss.item()
        
        pbar.set_description(
            f"t : {t:.3f}, ema loss (0.9) : {emaloss:.4f}, ema spec {ema_spec:.4f} ema bias (0.9) : {emabias:.4f}, lr {learning_rate(t,args.lr0):.4f}, sigma : {sigma(t,epoch):.4f}"
        )

        
        track[0].append(emaloss)
        track[1].append(ema_scale)
        track[2].append(ema_interp)
        track[3].append(emabias)
        track[4].append(ema_spec)

        if (idx%512)==0:
            
            print(f"scale : {scale.detach().cpu().numpy()}", flush=True)
            print(f"interp : {F.sigmoid(interp.detach().cpu()).numpy()}", flush=True)
        if (idx%512)==0 or idx==len(ensemble_dataset)-1:

            var_gen = torch.std(gen.detach(),dim=0, unbiased=True).cpu().numpy()
            var_real = torch.std(batch_y.detach(),dim=0, unbiased=True).cpu().numpy()

            n_var = 2 if args.convert_ff_t else 3
            fig, axs = plt.subplots(2,n_var,figsize = (6,6), sharex=True, sharey=True)
            for j in range(n_var):
                cmap = 'viridis' if (j< n_var - 1) else 'coolwarm'
                axs[0,j].imshow(var_real[j], origin = 'lower', cmap=cmap)
                axs[1,j].imshow(var_gen[j], origin = 'lower', cmap=cmap, vmin = var_real[j].min(), vmax = var_real[j].max())

            fig.tight_layout()
            plt.savefig(output_dir + f'std_real_vs_fake_{date}_{lt}_{idx}_{t:.3f}_{epoch}.png')
            plt.close()
            if not args.convert_ff_t:
                with torch.no_grad():
                    gen, batch_y = convert_uvt2fft(gen.detach(), batch_y.detach())
            ff_gen = gen[:,0].detach().cpu().mean(dim=0).numpy()
            ff_real = batch_y[:,0].detach().cpu().mean(dim=0).numpy()
            fig, axs = plt.subplots(1,2,figsize = (6,6), sharex=True, sharey=True)

            axs[0].imshow(ff_gen, origin = 'lower', cmap='viridis')
            axs[1].imshow(ff_real, origin = 'lower', cmap='viridis')

            fig.tight_layout()
            plt.savefig(output_dir + f'ff_mean_real_vs_fake_{date}_{lt}_{idx}_{t:.3f}_{epoch}.png')
            plt.close()

    fig, axs = plt.subplots(1,5,figsize = (16,4))
    axs[0].plot(track[0])
    for j in range(14):
        if j<=args.pca_cut:
            axs[1].plot(np.array(track[1])[:,j])
            axs[2].plot(np.array(track[2])[:,j])
        else:
            axs[1].plot(np.array(track[1])[:,j], linestyle='dashed')
            axs[2].plot(np.array(track[2])[:,j], linestyle='dashed')
    axs[3].plot(track[3])
    axs[4].plot(track[4])
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')
    fig.tight_layout()
    plt.savefig(output_dir + f'loss_scale_interp_bias.png')
    plt.close()
    np.save(output_dir + "ema_loss.npy",track[0])
    np.save(output_dir + "ema_scale.npy",track[1])
    np.save(output_dir + "ema_interp.npy",track[2])
    np.save(output_dir + "ema_bias.npy",track[3])
    np.save(output_dir + "ema_spec.npy",track[4])

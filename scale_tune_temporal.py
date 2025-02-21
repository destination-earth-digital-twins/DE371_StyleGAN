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
import utils.utils as utils

#from hyperparams.util import str2intlist, load_all_lt, select_random_dates, load_whole_model, list_all_obs
# Tunning alpha (theta) and beta (gamma) terms
from glob import glob

def add_noise(gamma,sig,device):
   noise = torch.empty(gamma.shape).normal_()
   return gamma + sig * noise.to(device)

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

if __name__=="__main__" :
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
    parser.add_argument("--optim_criterion", type=str, default='distrib_matching', choices=['distrib_matching','exchangeability'])


    ########################### Directories ###########################
    parser.add_argument("--fake_data_dir", type=str, 
                        default='/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3/Inversion_Perceptual_Random_VGG_Loss_sol3/')
    parser.add_argument("--real_data_dir", type=str, 
                        default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    parser.add_argument("--ensemble_data_dir", type=str, 
                        default='/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3/Pack_Perceptual_Random_VGG_Loss_sol3/')
    parser.add_argument("--ckpt_dir", type=str, 
                        default='/project/home/p200177/DE_371/resources/models/trained_generator/000024.pt')
    parser.add_argument("--eigendir", type=str, 
                        default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/eigenvalues_gan_training/')
    parser.add_argument("--output_dir", type=str, 
                        default='/project/scratch/p200177/DE_371/victorsanchez/results/scaled_perturbation/ScaleTune/')
    parser.add_argument('--scale_dir', type=str, default="")
    parser.add_argument('--scale_interp_step',type=int, default=-1)
    parser.add_argument("--leadtimes", type=utils.str2intlist, default=[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45])
    parser.add_argument("--dt",  type=int, default=3, help='time between two frames')


    args = parser.parse_args()


    output_dir = f"{args.output_dir}interp_scale_pca_{args.pca_cut}_{args.inflate_random}_{args.inflate}_bias_{args.start}_{args.lambda_bias}_spread_{args.lambda_spread}_ff_{args.convert_ff_t}_{args.invert_step}/"
    os.makedirs(output_dir, exist_ok=True)
    instances = len(glob(output_dir + "Instance_*/"))
    print("instances already existing", instances)
    os.makedirs(output_dir + f"Instance_{instances+1}/",exist_ok=True)
    output_dir = output_dir + f"Instance_{instances+1}/"
    df = pd.read_csv(args.real_data_dir + 'Large_lt_val_labels.csv') #Large_lt_val_labels
    df_date = df.copy()

    liste_dates = df_date['Date'].unique().tolist()
    print(liste_dates)
    # leadtimes = [6,12,18,24,30,36,42]

    ensemble_dataset = list(liste_dates)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    # Coefficient to generate first time step
    betas = torch.tensor(np.load(os.path.join(args.scale_dir,"ema_scale.npy")).astype(np.float32)[args.scale_interp_step], device=device)
    alphas = torch.tensor(np.load(os.path.join(args.scale_dir,"ema_interp.npy")).astype(np.float32)[args.scale_interp_step], device=device)

    if args.start=='ones':
        theta = torch.ones((14,),dtype=torch.float32, requires_grad=True,device=device)
        gamma = 0.05 * torch.ones((14,),dtype=torch.float32,device=device)
    elif args.start=='zeros':
        theta = torch.zeros((14,),dtype=torch.float32, requires_grad=True,device=device)
        gamma = -1.0 * torch.ones((14,),dtype=torch.float32,device=device, requires_grad=True)
    else:
        raise RuntimeError("Start unspecified")

    optimizer = optim.Adam([theta, gamma], lr = args.lr0)
    track = [[],[],[],[],[]]
    for epoch in range(args.n_epochs):
        shuffle(ensemble_dataset)
        print("#"*80)
        print(f"Epoch {epoch}")
        print("#"*80)
        pbar = tqdm(len(ensemble_dataset))
        for idx, date in enumerate(ensemble_dataset):
            leadtimes = args.leadtimes[np.random.choice(len(args.leadtimes)-2):]
            print(date, leadtimes)
            gen=None
            for id_lt, lt in enumerate(leadtimes):
                batch_w = torch.tensor(np.load(args.fake_data_dir + f"w_{date[:10]}_{lt}_{args.invert_step}.npy").astype(np.float32)).to(device)
                batch_y = torch.tensor(np.load(args.ensemble_data_dir + f"Rsemble_{date[:10]}_{lt}.npy").astype(np.float32)).to(device)
                batch_w_next = torch.tensor(np.load(args.fake_data_dir + f"w_{date[:10]}_{lt+args.dt}_{args.invert_step}.npy").astype(np.float32)).to(device)
                batch_y_next = torch.tensor(np.load(args.ensemble_data_dir + f"Rsemble_{date[:10]}_{lt+args.dt}.npy").astype(np.float32)).to(device)

                batch_delta_w = ( batch_w_next - batch_w )
                batch_delta_y = ( batch_y_next - batch_y )

                t =  idx / len(ensemble_dataset)
                optim.lr = learning_rate(t, args.lr0)
                gamma_noise = add_noise(gamma,sigma(t,epoch),device)
                theta_noise = add_noise(theta,sigma(t,epoch),device)

                try:
                    Delta_Cov = torch.load(args.fake_data_dir + f'Delta_Cov_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')
                    Delta_w_avg = torch.load(args.fake_data_dir + f'Delta_w_avg_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')
                    if id_lt == 0:
                        Cov = torch.load(args.fake_data_dir + f'Cov_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')
                        w_avg = torch.load(args.fake_data_dir + f'w_avg_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')

                    if Delta_w_avg.shape!=(args.pca_cut,512):
                        Delta_Cov, Delta_w_avg = pca.compute_K_covariance(batch_delta_w[:,:args.pca_cut],cut=args.n_samples-1)
                        torch.save(Delta_Cov, args.fake_data_dir + f'Delta_Cov_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')
                        torch.save(Delta_w_avg, args.fake_data_dir + f'Delta_w_avg_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')
                        if id_lt == 0:
                            Cov, w_avg = pca.compute_K_covariance(batch_w[:,:args.pca_cut],cut=args.n_samples-1)
                            torch.save(Cov, args.fake_data_dir + f'Cov_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')
                            torch.save(w_avg, args.fake_data_dir + f'w_avg_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')

                except FileNotFoundError:
                    Delta_Cov, Delta_w_avg  = pca.compute_K_covariance(batch_delta_w[:,:args.pca_cut],cut=args.n_samples-1)
                    torch.save(Delta_Cov, args.fake_data_dir + f'Delta_Cov_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')
                    torch.save(Delta_w_avg, args.fake_data_dir + f'Delta_w_avg_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')
                    if id_lt == 0:
                        Cov, w_avg  = pca.compute_K_covariance(batch_w[:,:args.pca_cut],cut=args.n_samples-1)
                        torch.save(Cov, args.fake_data_dir + f'Cov_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')
                        torch.save(w_avg, args.fake_data_dir + f'w_avg_{date[:10]}_{lt}_{args.pca_cut}_{args.invert_step}.pt')
                
                try :
                    assert Delta_w_avg.shape==(args.pca_cut,512)
                    if id_lt == 0:
                        assert w_avg.shape==(args.pca_cut,512)
                except AssertionError:
                    print(date, lt, batch_w.shape, Delta_w_avg.shape)
                    raise AssertionError("Uncorrect shape")
                
                if id_lt == 0:
                    gen = smpca.fast_style_mixing(
                        alphas=alphas,
                        betas=betas,
                        batch_w=batch_w,
                        K=Cov,
                        w_avg=w_avg,
                        n_samples=args.n_samples,
                        G=G,
                        Whitening=Whitening,
                        device=device,
                        beta_rule=args.scale_rule
                    )

                gen_next = smpca.fast_style_mixing_temporal(
                        dt=args.dt,
                        theta=theta,
                        gamma=gamma,
                        batch_w=batch_w,
                        batch_w_next=batch_w_next,
                        K=Delta_Cov,
                        w_avg=Delta_w_avg, 
                        n_samples=args.n_samples,
                        G=G,
                        Whitening=Whitening,
                        device=device,
                        beta_rule=args.scale_rule
                )
                
                batch_delta_gen = gen_next - gen

                if args.convert_ff_t:
                    gen, batch_y = convert_uvt2fft(gen, batch_y)
                
                # mean_loss = F.l1_loss(batch_delta_gen.mean(dim=0), batch_delta_y.mean(dim=0))
                # inflation = args.inflate if not args.inflate_random else (1.0 + uniform(0,args.inflate))
                # std_loss = F.l1_loss(torch.std(batch_delta_gen,dim=0, unbiased=True), inflation * torch.std(batch_delta_y,dim=0, unbiased=True))

                mean_loss = F.l1_loss(gen.mean(dim=0), batch_y.mean(dim=0))
                mean_loss += F.l1_loss(gen_next.mean(dim=0), batch_y_next.mean(dim=0))
                inflation = args.inflate if not args.inflate_random else (1.0 + uniform(0,args.inflate))
                std_loss = F.l1_loss(torch.std(gen,dim=0, unbiased=True), inflation * torch.std(batch_y,dim=0, unbiased=True))
                std_loss += F.l1_loss(torch.std(gen_next,dim=0, unbiased=True), inflation * torch.std(batch_y_next,dim=0, unbiased=True))

                if args.lambda_spectrum>0.0:
                    loss = args.lambda_bias * mean_loss + args.lambda_spread * std_loss# \

                
                else:
                    loss = args.lambda_bias * mean_loss + args.lambda_spread * std_loss
                

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update 
                gen = gen_next.clone().detach()

                emaloss = 0.9 * emaloss + 0.1 * loss.item() if idx>0 else loss.item()
                ema_spec = 0

                if args.scale_rule=='sigmoid':
                    ema_scale = 0.1 * F.sigmoid(gamma.detach().cpu()).numpy() + 0.9 * ema_scale if idx>0 else F.sigmoid(gamma.detach().cpu()).numpy()
                else:
                    ema_scale = 0.1 * gamma.detach().cpu().numpy() + 0.9 * ema_scale if idx>0 else gamma.detach().cpu().numpy()
                ema_interp = 0.1 * F.sigmoid(theta).detach().cpu().numpy() + 0.9 * ema_interp if idx>0 else F.sigmoid(theta).detach().cpu().numpy()

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
                    
                    print(f"gamma : {gamma.detach().cpu().numpy()}", flush=True)
                    print(f"theta : {F.sigmoid(theta.detach().cpu()).numpy()}", flush=True)
                if (idx%512)==0 or idx==len(ensemble_dataset)-1:

                    var_gen = torch.std(batch_delta_gen.detach(),dim=0, unbiased=True).cpu().numpy()
                    var_real = torch.std(batch_delta_y.detach(),dim=0, unbiased=True).cpu().numpy()

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
                            batch_delta_gen, batch_delta_y = convert_uvt2fft(batch_delta_gen.detach(), batch_delta_y.detach())
                    ff_gen = batch_delta_gen[:,0].detach().cpu().mean(dim=0).numpy()
                    ff_real = batch_delta_y[:,0].detach().cpu().mean(dim=0).numpy()
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

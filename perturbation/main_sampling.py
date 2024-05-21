#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:21:37 2023

@author: brochetc

Main pod sampling script

"""
import smpca
import torch
import argparse
from stylegan2 import Generator
import os
import numpy as np
import pivotal as pit
import matplotlib.pyplot as plt
import pickle
import metrics4ensemble as metrics
from time import perf_counter
import perturbation.utils as utils

device = 'cuda:0'

def str2list(li):
    if type(li)==list:
        li2 = li
        return li2
    
    elif type(li)==str:
        li2=li[1:-1].split(',')
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))


def compute_generate_save(i,j, N_samples,G, args, n_mean) :
    
    Ens_r = torch.tensor(np.load(args.real_data_dir + 'Rsemble_{}_{}.npy'.format(float(i),float(j))).astype(np.float32))
    
    z = torch.empty((n_mean, 512)).normal_().to(device)
    
    latent_mean = G.style(z).mean(dim=0)
    
    index = i * 8 + j

    t0 = perf_counter()

    w_ens, noises, G_opt = pit.pitoptimize(Ens_r, G, latent_mean, device, args, index = index)
    
    t1 = perf_counter() - t0
    
    print('elapsed in optimize', t1)
    
    t2 = perf_counter()
    
    gen, w_new = smpca.sm_pca(w_ens, G_opt, 
                         N_samples, 
                         args.style_indices, device, args.sample_rule)
    
    t3 = perf_counter()-t2
    print('elapsed in perturbing', t3)
    
    np.save(args.output_dir + 'Fsemble_{}_{}.npy'.format(i,j), gen)
    
    return 0


def compute_generate_analyze_save(i,j, N_samples, G, args, n_mean, metrics_list, option = None):
    
    Ens_r = torch.tensor(np.load(args.real_data_dir + 'Rsemble_{}_{}.npy'.format(float(i),float(j))).astype(np.float32))
    
    z = torch.empty((n_mean, 512)).normal_().to(device)
    
    latent_mean = G.style(z).mean(dim=0)
    
    index = i * 8 + j
    
    print('############### Inverting ###############')

    if option=='load' :

        w_ens, noises, G_opt = pit.pitoptimize(Ens_r, G, latent_mean, device, args, index = index)
    else :
        
        w_ens = torch.tensor(np.load(args.data_dir + 'w_{}_{}.npy'.format(float(i),float(j))).astype(np.float32))
        
        G_opt = G

    print('############### Perturbating ###############')
          
    gen, w_new = smpca.sm_pca(w_ens, G_opt, 
                         N_samples, 
                         args.style_indices, device, args.sample_rule)
    
    gen = utils.rescale(gen, args.Mean, args.Maxs, args.scale)
    Ens_r = utils.rescale(Ens_r.detach().cpu().numpy(), args.Mean, args.Maxs, args.scale)
    
    print('############### Evaluating metrics ###############')
    
    dic_metrics = {}
    
    for m in metrics_list :
        
        print(m)
        
        metr = getattr(metrics,m)
        
        if m in metrics.standalone_metrics :
        
            dic_metrics[m] = metr(gen)
            
        elif m in metrics.distance_metrics :
            
            dic_metrics[m] = metr(Ens_r, gen)
            
        else:
            
            raise ValueError('Metric unknown')
        
    np.save(args.output_dir + 'Fsemble_{}_{}.npy'.format(i,j), gen)
    pickle.dump(dic_metrics,open(args.output_dir + 'metrics_{}_{}.p'.format(i,j), 'wb'))
    
    
    

def analyzeNoiseimpact(i,j, G, args, n_mean):
    
    Ens_r = torch.tensor(np.load(args.real_data_dir + 'Rsemble_{}_{}.npy'.format(float(i),float(j))).astype(np.float32))[0:1]
    
    print('Ens_r shape', Ens_r.shape)
    
    z = torch.empty((n_mean, 512)).normal_().to(device)
    
    latent_mean = G.style(z).mean(dim=0)
    
    index = i * 8 + j

    w_ens, noises, G_opt = pit.pitoptimize(Ens_r, G, latent_mean, device, args, index = index)
    
    print("######### Analyzing impact of noise ###########")
    
    generated = []
    
    with torch.no_grad():
    
        img_gen_noise, _ = G_opt([w_ens], input_is_latent=True, noise = noises)
        
        generated.append(img_gen_noise.detach().cpu().numpy())
    
        for n in range(n_mean) :
                
            imgen, _ = G_opt([w_ens], input_is_latent=True)
            
            generated.append(imgen.detach().cpu())
            
    generated = np.concatenate(generated, axis = 0)
    
    generated = utils.rescale(generated, args.Mean, args.Maxs, args.scale)
    
    Ens_check = utils.rescale(Ens_r.numpy(), args.Mean, args.Maxs, args.scale)
    
    diff_mean = (generated[0]-generated[1:].mean(axis=0))
    
    variance = generated.std(axis=0)
    
    error = np.abs((generated[1:] - Ens_check)).mean(axis=0)
    
    error_inv = np.abs((generated[0:1] - Ens_check)).mean(axis=0)    
    
    for i in range(3):
        
        CMAP = 'viridis' if i<2 else 'coolwarm'
        
        """plt.imshow(generated[0][i], origin = 'lower', cmap = CMAP)
        plt.colorbar()
        plt.savefig(args.output_dir + 'with_fixed_noise_{}_{}.png'.format(i, index))
        plt.close()
        plt.imshow(generated[1][i], origin = 'lower', cmap = CMAP)
        plt.colorbar()
        plt.savefig(args.output_dir + 'with_random_noise_{}_{}.png'.format(i, index))
        plt.close()
        
        plt.imshow(diff_mean[i], origin = 'lower', cmap = CMAP)
        plt.colorbar()
        plt.savefig(args.output_dir + 'diff_mean_{}_{}.png'.format(i, index))
        plt.close()
        
        
        plt.imshow(diff_mean[i], origin = 'lower', cmap = CMAP)
        plt.colorbar()
        plt.savefig(args.output_dir + 'diff_mean_{}_{}.png'.format(i, index))
        plt.close()"""
        
        plt.imshow(error[i], origin = 'lower', cmap = CMAP)
        plt.colorbar()
        plt.savefig(args.output_dir + 'noise_induced_error_{}_{}.png'.format(i, index))
        plt.close()
            
        plt.imshow(error_inv[i], origin = 'lower', cmap = CMAP)
        plt.colorbar()
        plt.savefig(args.output_dir + 'inversion_induced_error_{}_{}.png'.format(i, index))
        plt.close()
        
        plt.imshow(np.abs(error_inv[i] - error[i]), origin = 'lower', cmap = CMAP)
        plt.colorbar()
        plt.savefig(args.output_dir + 'inversion_to_noise_error_{}_{}.png'.format(i, index))
        plt.close()
    
    return 0

def generate_save(lt, index, w_ens, N_samples, G, args, device ='cuda:0', option = 'load'):

    w_new = torch.tensor(w_ens, dtype = torch.float32)
    
    assert torch.isfinite(w_new).all()
    
    Nbatches = N_samples // 256
    
    gen_big = np.zeros((875,3,128,128))
    
    for b in range(Nbatches +1) :
        print('batch {}'.format(b))
        with torch.no_grad() : 
            gen, _ = G([w_new[b * 256, (b+1) * 256].to(device)], input_is_latent = True)
        
        gen_big[b * 256 : (b+1)* 256] = gen.detach().cpu().numpy()
        
    np.save(args.output_dir + 'Pert_grand_ensemble_{}_{}.npy'.format(lt, index), gen_big)
    
    return 0


if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ckpt_dir', type = str, 
                        default ='/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/Database_latent/285000.pt')
    parser.add_argument('--data_dir', type = str, 
                        default ='/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/Database_latent/')
    parser.add_argument('--real_data_dir', type = str, 
                        default ='/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/Database_latent/')
    parser.add_argument('--output_dir',type = str, 
                        default ='/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/Database_latent/')
    
    parser.add_argument(
        "--ckpt", type=str, help="path to the model checkpoint",
        default='/scratch/mrmn/brochetc/GAN_2D/projection_expe/285000.pt'
    )
    
    parser.add_argument(
        "--size", type=int, default=128, help="output image sizes of the generator"
    )
    
    parser.add_argument(
        "--reg_strength",
        type=float,
        default=1.0,
        help=" ball holder regularization strength",
    )
       
    parser.add_argument(
        "--ball_holder_interv",
        type=int, default=50,
        help="interval of regularization of g with ball holder",
    )
    
    parser.add_argument(
        "--morph_alpha",
        type=float,
        default=30,
        help=" ball holder motion for morphing",
    )
       
    parser.add_argument(
        "--num_fix",
        type=int, default=10,
        help="number of fixed points for ball_holder",
    )
    
    
    ############################ PIVOTAL INVERSION PARAMETERS #################
    
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float, 
        default=0.25,
        help="duration of the learning rate decay",
    )
    
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    
    parser.add_argument("--lr_pit", type=float, default=0.001, 
                        help="Generator tuning learning rate")

    parser.add_argument(
        "--noise", type=float, default=0.005, help="strength of the noise level"
    )
    
    
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    
    parser.add_argument("--invstep", type=int, default=300, help="optimize iterations")
    parser.add_argument("--pitstep", type=int, default=300, help="pivotal iterations")
    
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization (inversion)",
    )
    parser.add_argument("--mse", type=float, default=1.0, help="weight of the mse loss")
    
    
    ######################## PERTURBATION PARAMETERS #######################
    
    parser.add_argument('--sample_rule', type=str, default='throughB', 
                        choices = ['random', 'silvermans', 'throughB'])
    parser.add_argument('--style_indices', type = str2list, default='[0,2,9,12]')
    
    
    parser.add_argument('--startdate', type = int, default=0)
    ###########################################################################
    ###########################################################################
    
    args = parser.parse_args()
    
    real_data_name = 'W_plus/samples/' #if args.space=='w_plus' else 'W/samples/'
    
    args.real_data_dir = args.data_dir + real_data_name
    
    data_name = 'W_plus/samples/'
    
    args.data_dir = args.data_dir + data_name
    
    args.output_dir = args.output_dir +  \
    f'{args.sample_rule}_{args.style_indices}_{args.invstep}_{args.pitstep}_{args.lr}_{args.lr_pit}_{args.reg_strength}_{args.ball_holder_interv}_{args.morph_alpha}_{args.num_fix}_freeNoise/' 
    
    print(args.output_dir)
    
    ################## carrying scaling info to pass it whenever needed
    
    args.Mean = np.load(args.data_dir + 'mean_with_orog.npy')[1:4].reshape(1,3,1,1)

    args.Maxs = np.load(args.data_dir + 'max_with_orog.npy')[1:4].reshape(1,3,1,1)
    
    args.scale = (1/0.95)
    
    ############################################################
    
    if not os.path.exists(args.output_dir) :
        os.mkdir(args.output_dir)
        os.mkdir(args.output_dir + '/samples/')
        os.mkdir(args.output_dir + '/log/')
    
    N_samples = 120
    
    G = Generator(128,512,8)
    
    G.load_state_dict(torch.load(args.ckpt_dir, map_location=torch.device('cpu'))['g_ema'])
    
    G = G.to(device)
    G.eval()
    
    metrics_list = ['quantiles', 'variance', 'std_diff', 'mean_bias']
    
    for i in range(args.startdate,74) : 
        
        for j in range(8) :
            
            print(i,j)
            
            compute_generate_analyze_save(i,j,120, G, args, 1000, metrics_list)

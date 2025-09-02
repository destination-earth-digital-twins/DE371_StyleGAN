#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:13:57 2023

@author: brochetc

The aim is to analyze the effect of intra-ensemble latent uncertainty
on the real space variability

"""
import torch
import torch.linalg as linalg
import perturbation.pca_stylegan as pca
import matplotlib.pyplot as plt
from gan.model.stylegan2 import Generator
from collections import OrderedDict
import numpy as np

def computeColoring(G, N_samples = 10000, verbose=True, device='cuda:0'):
    """
    G : nn.Module to generate w samples
    N_samples :  number of samples to estimate covariance matrix
    """

    z = torch.empty(N_samples, 512).normal_().to(device)

    with torch.no_grad():
        w = G.style(z)
        w = w #* torch.rsqrt(torch.linalg.norm(w, dim=-1, keepdims=True) + 1e-8)

    w0 = w.mean(dim=0)
    Ens = w - w0

    sigmas, q = pca.ensemble_pod(Ens, cut=512, verbose=True)
    print("q shape", q.shape)
    print("q id_test", (q.squeeze() @ q.squeeze().t()))

    #sigmas = torch.sqrt(torch.abs(sigmas))
    sigmas = torch.sqrt(torch.abs(sigmas))

    Coloring = (q.squeeze() @ torch.diag_embed(sigmas).squeeze()) @ q.squeeze().t()

    if verbose:

        Ens_t = Ens.t()
        Ens = Ens

        M = Ens_t @ Ens * (1 / (N_samples -1))

        print("cov shape", M.shape)
        id_test = Coloring @ Coloring - M.squeeze()
        print("id test coloring", torch.linalg.norm(id_test), torch.max(torch.abs(id_test)), torch.min(torch.abs(id_test)))

    return Coloring, sigmas.squeeze(), q, w0


def AkaikeRank(sigmas):

    lambdas = sigmas / sigmas.sum()
    print("lambdas", lambdas)

    rank = np.sum(np.exp(-lambdas * np.log(lambdas)))

    return rank

def test_white(G, Whitening, w0=None,N_samples=10000, device='cuda:0'):

    z = torch.empty(N_samples,512).normal_().to(device)

    with torch.no_grad():
        w = G.style(z)
        w = w #* torch.rsqrt(torch.linalg.norm(w, dim=-1, keepdims=True) + 1e-8)

    if w0 is None:
        w0 = w.mean(dim=0)
    Ens = w - w0
    print(Ens.shape)

    wh = torch.bmm(Whitening.unsqueeze(0).repeat(N_samples,1,1), Ens.unsqueeze(-1)).squeeze()

    Cov = wh.t() @ wh * (1.0 / (N_samples - 1))

    id_test = Cov - torch.eye(512).to(device)

    print(torch.linalg.norm(id_test), torch.max(torch.abs(id_test)), torch.min(torch.abs(id_test)))


if __name__=="__main__":

    ckpt_dir = '.pt'
    output_dir = '/Eigenvalues/'
    device = 'cuda:0'

    G = Generator(256, 512,n_mlp=8,nb_var=4)

    #print('###########################################"##################################################################################################################')
    ckpt = torch.load(ckpt_dir, map_location='cpu')['g_ema']

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

    print('Computing Coloring')
    Coloring, sigmas, q, w0t = computeColoring(G, N_samples=20000)
    print('Coloring.shape', Coloring.shape)
    print('Computing Akaike rank')
    rank = AkaikeRank(sigmas.cpu().numpy())
    print(rank, int(rank))

    if rank <=512:
        epsilon = sigmas[int(rank):].sum()
        eps = epsilon.cpu().numpy()
    else:
        epsilon = torch.tensor(0.0).to(device)
        eps = 0.0
    sig = sigmas.cpu()

    print("epsilon", eps, 
        "sigmas", sig.max(), sig.min(), sig.mean())

    plt.plot(np.arange(512),np.log(sig))
    if rank <512:
        plt.hlines(eps, 0,512)
    plt.grid()
    plt.savefig(output_dir + 'w_eigenvalues.png')
    plt.close()

    sigmas_inv = sigmas / (sigmas ** 2 + epsilon ** 2)
    print('inv shape', sigmas_inv.shape)
    plt.plot(np.log(sigmas_inv.cpu().numpy()), label = 'corrected')
    plt.plot(np.log(1.0 / sig), label = 'raw')
    plt.legend()
    if rank <512:
        plt.hlines(eps, 0,512)
    plt.grid()
    plt.savefig(output_dir + 'w_invert_eigenvalues.png')
    plt.close()

    Whitening = (q.squeeze() @ torch.diag_embed(sigmas_inv.squeeze())) @ q.squeeze().t()
    print("Whitening shape", Whitening.shape)
    print("Coloring shape", Coloring.shape)
    id_test = Coloring @ Whitening - torch.eye(512).to(device)
    
    print("Invert Coloring and Whitening : \t", torch.linalg.norm(id_test), torch.max(torch.abs(id_test)))
    plt.plot(torch.diag((Coloring @ Whitening).cpu(),0))
    if rank <512:
        plt.hlines(eps, 0,512)
    plt.grid()
    plt.savefig(output_dir + 'test_eye.png')
    plt.close()

    torch.save(Whitening.cpu(),output_dir + 'Whitening.pt')
    torch.save(Coloring.cpu(),output_dir + 'Coloring.pt')
    torch.save(w0t.cpu(),output_dir + 'latent_mean.pt')

    for n in [2,10,20,50,100,1000,10000,20000]:
        print(f'testing decorrelation N_samples {n}')
        test_white(G, Whitening, w0=w0t, N_samples=n)


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
from torch.autograd.functional import vjp, jvp


def ensemble_pod(Ens, cut, verbose=False):
    """
    
    Compute the first (= highest eigenvalues) eigenvectors of the Ens covariance matrix
    
    Ens : B x N x D torch.tensor where N is the number of samples and D the dimension
    of the space where they live
    
    cut : int
    
    """
    
    if verbose: print(Ens.shape)
    
    size = Ens.shape[1] if Ens.ndim==3 else Ens.shape[0]
    if verbose: print("size", size)
    
    Dim = Ens.shape[-1]
    
    if Ens.ndim==3 :
    
        Ens_t = Ens.permute((0,2,1))
                
    else :
        
        Ens_t = Ens.t().unsqueeze(0)
        Ens = Ens.unsqueeze(0)

    if verbose : 
        print("(transpose) ensemble shape", Ens_t.shape, Ens.shape)
    
    M = torch.bmm(Ens_t, Ens) * (1 / (size -1))

    if verbose: print("empirical cov shape", M.shape)

    d, q  = linalg.eigh(M)
    
    if verbose:
        print("remaining values on 0th index", d[0,Dim-cut:])
    
    d[:,:Dim-cut] = torch.zeros_like(d[:, :Dim-cut])
    
    if verbose:
        print("max diag", d.max())
    
    return d, q

def computeReducedCovariance(Ens_w, cut, verbose=False,device='cuda:0'):
    N, R, D = Ens_w.shape # Number, Repeats, Dimension (typicallly = 16, 14, 512)
    w_avg = Ens_w.mean(dim=0)
    Ens_0 = (Ens_w.contiguous() - w_avg).view(R,N,D)
    
    sigmas, q = ensemble_pod(Ens_0,cut=cut, verbose=verbose)
    sigmas = torch.sqrt(torch.abs(sigmas))
    sigmas = torch.diag_embed(sigmas)
    
    if verbose:
        print("Max, min values, q shape", sigmas.max(), sigmas.min(), q.shape)
    
    assert torch.isfinite(q).all()

    if verbose:
        id_test = torch.bmm(q, q.permute(0,2,1)) - torch.eye(q.shape[-1]).to(device)
        print('id_test linalg norm', torch.linalg.norm(id_test),torch.max(torch.abs(id_test)))

    Cov = torch.bmm(q,torch.bmm(sigmas, q.permute(0,2,1))).contiguous().to(device)
    if verbose: print("Covariance matrix shape", Cov.shape)

    return Cov, w_avg

def computeReducedCovarianceW(Ens_w,cut,Whitening,Coloring,verbose=False,device='cuda:0', renorm=False):
    N, R, D = Ens_w.shape # Number, Repeats, Dimension (typicallly = 16, 14, 512)
    EnsNorm = torch.linalg.norm(Ens_w, dim=-1, keepdims=True)
    #Ens_w1 = Ens_w * torch.rsqrt(EnsNorm + 1e-8)
    w_avg = Ens_w.mean(dim=0)

    if verbose:
        print("Ensemble w shape", Ens_w.shape, w_avg.shape)

    Ens_0 = torch.bmm(Whitening.to(device).unsqueeze(0).repeat(R * N,1,1),(Ens_w - w_avg).view(R * N,D).unsqueeze(-1)).view(R,N,D)

    if verbose:
        print("Whitened ensemble shape", Ens_0.shape)

    sigmas, q = ensemble_pod(Ens_0, cut=cut, verbose=verbose)
    if verbose:
        print("Sigmas and q shape", sigmas.shape, q.shape)
    sigmas = torch.sqrt(torch.abs(sigmas))
    sigmas = torch.diag_embed(sigmas) / torch.max(sigmas) if renorm else torch.diag_embed(sigmas)
    
    if verbose:
        print("Max, min values, q shape",sigmas.max(), sigmas.min(), q.shape)
    
    assert torch.isfinite(q).all()
    if verbose:
        id_test = torch.bmm(q, q.permute(0,2,1)) - torch.eye(q.shape[-1]).to(device)
        print('id_test linalg norm', torch.linalg.norm(id_test),torch.max(torch.abs(id_test)))

    Cov = torch.bmm(q,torch.bmm(sigmas, q.permute(0,2,1))).contiguous() # Cov of shape R x D x D
    if verbose:
        print("Covariance shape", Cov.shape)
        print("Coloring shape", Coloring.shape)
    Cov = torch.bmm(Coloring.unsqueeze(0).to(device).repeat(R,1,1), Cov) # Cov of shape R x D x D
    return Cov, w_avg
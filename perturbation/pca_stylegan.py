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
    r"""
    Compute the first (= highest eigenvalues) eigenvectors of the Ens covariance matrix
    
    Inputs : 
        - Ens : B x N x D (torch.tensor) where N is the number of samples and D the dimension
            of the space where they live

        - cut : (int)
    
    Outputs :
        - eigenvalues : (torch.tensor)
        
        - eigenvectors : (torch.tensor)
        
    Note : pod = Proper orthogonal decomposition
    """
    
    if verbose: print(Ens.shape)
    
    size = Ens.shape[1] if Ens.ndim==3 else Ens.shape[0]
    if verbose: print("size", size)
    
    Dim = Ens.shape[-1]
    
    if Ens.ndim==3 :
    
        Ens_t = Ens.permute((0,2,1)) # Ensemble Transpose
                
    else :
        
        Ens_t = Ens.t().unsqueeze(0)  # Ensemble Transpose
        Ens = Ens.unsqueeze(0)

    if verbose : 
        print("(transpose) ensemble shape", Ens_t.shape, Ens.shape)
    
    cov_matrix = torch.bmm(Ens_t, Ens) * (1 / (size -1))

    if verbose: print("empirical cov shape", cov_matrix.shape)

    eigenvalues, eigenvectors  = linalg.eigh(cov_matrix)
    
    if verbose:
        print("remaining values on 0th index", eigenvalues[0,Dim-cut:])
    
    eigenvalues[:,:Dim-cut] = torch.zeros_like(eigenvalues[:, :Dim-cut])
    
    if verbose:
        print("max diag", eigenvalues.max())
    
    return eigenvalues, eigenvectors

def computeReducedCovariance(Ens_w, cut, verbose=False, device='cuda:0'):
    r"""
    Compute a reduced version of the covariance matrix
    
    Inputs : 
        - Ens_w : N x R x D (torch.tensor) # Number, Repeats, Dimension (typicallly = 16, 14, 512)

        - cut : (int)

        - verbose (bool) default=False
        
        - device (str) default='cuda:0'
    
    Outputs :
        - Cov (torch.tensor) : Reduced covariance matrix
        
        - w_avg : (torch.tensor) : Mean of Ens_w along dim=0

    """

    # Normalizing Ens_w
    number, repeats, dimension = Ens_w.shape 
    w_avg = Ens_w.mean(dim=0)
    Ens_0 = (Ens_w.contiguous() - w_avg).view(repeats,number,dimension)
    
    # POD on the normalized Ens_w
    sigmas, q = ensemble_pod(Ens_0, cut=cut, verbose=verbose)
    sigmas = torch.sqrt(torch.abs(sigmas))
    sigmas = torch.diag_embed(sigmas) # Creates a diagonal matrix containing the eigenvalues
    
    if verbose:
        print("Max, min values, q shape", sigmas.max(), sigmas.min(), q.shape)
    
    assert torch.isfinite(q).all() # All eignenvalues must be finite

    if verbose:
        id_test = torch.bmm(q, q.permute(0,2,1)) - torch.eye(q.shape[-1]).to(device)
        print('id_test linalg norm', torch.linalg.norm(id_test),torch.max(torch.abs(id_test)))

    # Reduced Covariance matrix
    Cov = torch.bmm(q,torch.bmm(sigmas, q.permute(0,2,1))).contiguous().to(device)
    if verbose: print("Covariance matrix shape", Cov.shape)

    return Cov, w_avg

def computeReducedCovarianceW(Ens_w, cut, verbose=False, device='cuda:0', renorm=False):
    r"""
    Compute a reduced version of the covariance matrix
    
    Inputs : 
        - Ens_w : N x R x D (torch.tensor) # Number, Repeats, Dimension (typicallly = 16, 14, 512)

        - cut : (int)

        - verbose (bool) default=False
        
        - device (str) default='cuda:0'
    
    Outputs :
        - Cov (torch.tensor) : Reduced covariance matrix
        
        - w_avg : (torch.tensor) : Mean of Ens_w along dim=0

    """

    # Computation of the mean of the ensemble w
    number, repeats, dimension = Ens_w.shape # Number, Repeats, Dimension (typicallly = 16, 14, 512)
    EnsNorm = torch.linalg.norm(Ens_w, dim=-1, keepdims=True)
    #Ens_w1 = Ens_w * torch.rsqrt(EnsNorm + 1e-8)
    w_avg = Ens_w.mean(dim=0)

    if verbose:
        print("Ensemble w shape", Ens_w.shape, EnsNorm.shape)

    Ens_0 =(Ens_w - w_avg).view(repeats,number,dimension)

    # POD on the normalized Ens_w
    sigmas, q = ensemble_pod(Ens_0, cut=cut, verbose=verbose)
    if verbose:
        print("Sigmas and q shape", sigmas.shape, q.shape)

    # Creates a diagonal matrix containing the eigenvalues
    sigmas = torch.sqrt(torch.abs(sigmas))
    sigmas = torch.diag_embed(sigmas) / torch.max(sigmas) if renorm else torch.diag_embed(sigmas)
    
    if verbose:
        print("Max, min values, q shape",sigmas.max(), sigmas.min(), q.shape)
    
    assert torch.isfinite(q).all() # All eignenvalues must be finite
    if verbose:
        id_test = torch.bmm(q, q.permute(0,2,1)) - torch.eye(q.shape[-1]).to(device)
        print('id_test linalg norm', torch.linalg.norm(id_test),torch.max(torch.abs(id_test)))

    Cov = torch.bmm(q,torch.bmm(sigmas, q.permute(0,2,1))).contiguous() # Cov of shape R x D x D
    return Cov, w_avg
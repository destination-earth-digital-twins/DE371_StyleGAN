import torch
import torch.nn.functional as F

import numpy as np


def cosineLoss(w, batch_w, verbose=False):
    """
    compute cosine similarity between samples from w and batch_w
    """

    sim = F.cosine_similarity(w,batch_w, dim=-1)
    if verbose:
        print(sim.shape)
        print(sim.mean(),sim.max())
    return 1.0 - sim
    

def recallWDistrib(N_samples, batch_w, G, loss, device ='cpu'):
    """
    Draw N_samples per sample in the batch_w tensor from a random normal distribution
    process these N_samples through the MLP of a stylegan generator G, giving N_sample styles
    finally select for each sample of batch_w the closest processed style (in the sense of the given loss)

    Inputs:
        N_samples :  int, number of samples to create for each given sample in batch_w

        batch_w : Tensor, shape B x C x D where C, D are the representation's dimension and B the batch size, assumed to be already on-device

        G :  StyleGAN-like generator, assumed to be already on-device

        loss : a minimisable distance function (not necessarily differentiable)

    Returns:

        batch_w_z : Tensor of same shape as batch_w (B x C x D)

    """

    B, C, D = batch_w.shape

    z = torch.empty(N_samples, B, C, D).normal_().view(-1,D).contiguous().to(device)

    assert (z.device==batch_w.device)
    print(z.device)

    with torch.no_grad():
        w = G.style(z).view(N_samples, B, C, D)
    print(w.shape)
    diff = loss(w, batch_w, verbose=True) # is of shape N_samples x B x C
    mins, argmins = torch.min(diff, dim=0, keepdim=True) # argmins is of shape 1 x B x C
    argmins = argmins.unsqueeze(-1).repeat(1,1,1,D) # argmins of shape 1 x B x C x D
    print(argmins.shape)
    batch_w_z = torch.gather(w,0,argmins).squeeze() # getting the w's that corresponds to mins of loss
    print(batch_w_z.shape)

    return batch_w_z, mins


def recallWDistribForLoop(N_samples, batch_w, G, loss, device ='cpu'):
    """
    Draw N_samples per sample in the batch_w tensor from a random normal distribution
    process these N_samples through the MLP of a stylegan generator G, giving N_sample styles
    finally select for each sample of batch_w the closest processed style (in the sense of the given loss)

    Use a for loop to retrieve minima (only use in checks and tests)

    Inputs:
        N_samples :  int, number of samples to create for each given sample in batch_w

        batch_w : Tensor, shape B x C x D where C, D are the representation's dimension and B the batch size, assumed to be already on-device

        G :  StyleGAN-like generator, assumed to be already on-device

        loss : a minimisable distance function (not necessarily differentiable)

    Returns:

        batch_w_z : Tensor of same shape as batch_w (B x C x D)

    """

    B, C, D = batch_w.shape

    z = torch.empty(N_samples, B, C, D).normal_().view(-1,D).contiguous().to(device)

    assert (z.device==batch_w.device)
    with torch.no_grad():
        w = G.style(z).view(N_samples, B, C, D)

    diff = loss(w, batch_w) # is of shape N_samples x B x C
    print(diff.shape)
    mins, argmins = torch.min(diff, dim=0, keepdim=True) # diff_min is of shape 1 x B x C
    print(mins.shape, mins.min())
    batch_w_z = torch.zeros((B,C,D))
    print(argmins.shape)
    print(batch_w_z.shape)

    for batch_idx in range(B):
        for chan_idx in range(C):
            batch_w_z[batch_idx, chan_idx] = w[argmins[0,batch_idx,chan_idx], batch_idx, chan_idx]

    return batch_w_z, mins
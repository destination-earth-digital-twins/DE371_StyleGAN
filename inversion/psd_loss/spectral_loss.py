#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
# from scipy.fftpack import dct, fft, idct
from scipy.signal import welch, coherence
import torch

class SpectralLoss(torch.nn.Module):
    def __init__(self):
        r''' Class that computes the Spectral Loss from https://arxiv.org/pdf/2501.19374

         '''
        super(SpectralLoss, self).__init__()


    # def compute_MSE_loss_from_PSD_decomposition(self, x,y):
    #     channels = x.shape[1]
    #     loss = 0
    #     for c in range(channels):
    #         PSD_x = welch(x[:, c : c + 1, :, :])
    #         PSD_y = welch(y[:, c : c + 1, :, :])
    #         Coh = coherence(x[:, c : c + 1, :, :],y[:, c : c + 1, :, :])
    #         loss += PSD_x + PSD_y - 2 * np.sqrt(PSD_x*PSD_y)*Coh

    #     return loss

    def compute_MSE_loss_from_PSD_adjusted_decomposition(self, x,y):
        channels = x.shape[1]
        loss = 0
        for c in range(channels):
            PSD_x = welch(x[:, c : c + 1, :, :].flatten(), fs=10000)[1]
            # print('PSD_x.shape', PSD_x.shape)
            PSD_y = welch(y[:, c : c + 1, :, :].flatten(), fs=10000)[1]
            # print('PSD_y.shape', PSD_y.shape)
            Coh = coherence(x[:, c : c + 1, :, :].flatten(),y[:, c : c + 1, :, :].flatten(), fs=10000)[1]
            # print('Coh.shape', Coh.shape)
            spectral_loss = np.sum((np.sqrt(PSD_x) - np.sqrt(PSD_y))**2 + 2 * np.sqrt(PSD_x*PSD_y)*(1-Coh))
            loss += spectral_loss 

        return loss
    
    def compute_AMSE_loss_from_PSD_adjusted_decomposition(self, x,y):
        channels = x.shape[1]
        loss = 0
        for c in range(channels):
            PSD_x = welch(x[:, c : c + 1, :, :].flatten(), fs=1)[1]
            # print('PSD_x.shape', PSD_x.shape)
            PSD_y = welch(y[:, c : c + 1, :, :].flatten(), fs=1)[1]
            # print('PSD_y.shape', PSD_y.shape)
            Coh = coherence(x[:, c : c + 1, :, :].flatten(),y[:, c : c + 1, :, :].flatten(), fs=1)[1]
            # print('Coh.shape', Coh.shape)
            spectral_loss =  np.sum((np.sqrt(PSD_x) - np.sqrt(PSD_y))**2 + 2 * np.max((PSD_x, PSD_y))*(1-Coh))
            loss += spectral_loss
            
        return loss

    def forward(self, x, y):
        return self.compute_AMSE_loss_from_PSD_adjusted_decomposition(x.cpu().detach().numpy(),y.cpu().detach().numpy())


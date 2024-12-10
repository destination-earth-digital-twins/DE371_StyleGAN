
"""
This script compares the distance between ensemble forecast members with different distance criterions
"""
import torch
import argparse
import os
import numpy as np
import torch.nn.functional as F
import perturbation.utils as utils
import matplotlib.pyplot as plt
from inversion.perceptual_loss.perceptual_loss import PerceptualLoss
from inversion.experimental_loss.ssim import MS_SSIM
import scipy
torch.manual_seed(42) #reproducibility of runs

def calc_anomaly_correlation_coefficient(x,y):
    # r"""Pearson product-moment correlation coefficient.

    # A measure of the linear association between the forecast and verification data that
    # is independent of the mean and variance of the individual distributions. This is
    # also known as the Anomaly Correlation Coefficient (ACC) when correlating anomalies.

    # .. math::
    #     corr = \frac{cov(f, o)}{\sigma_{f}\cdot\sigma_{o}},

    # where :math:`\sigma_{f}` and :math:`\sigma_{o}` represent the standard deviation
    # of the forecast and verification data over the experimental period, respectively.
    
    # Args:

    # """
    #return torch.cov(torch.cat((x.unsqueeze(1),y.unsqueeze(1)), dim=1)) / (torch.std(x,dim=0, unbiased=True)*torch.std(y,dim=0, unbiased=True))
    
    #ACC computation without m
    # https://confluence.ecmwf.int/display/FUG/Section+12.A+Statistical+Concepts+-+Deterministic+Data
    num = (x*y).mean()
    square_denom = (x**2).mean()*(y**2).mean()
    # averageing on batch samples
    res = torch.mean(num / torch.sqrt(square_denom), dim=0)
    return res

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################
    # Checkpoint directory - PATH to generator's weight
    parser.add_argument('--ckpt_dir', type = str, 
                        default ='/project/home/p200177/DE_371/resources/models/trained_generator/000024.pt')
    # Real Data Directory - PATH to samples of the dataset
    
    parser.add_argument('--pack_dir', type = str, default ='/project/home/p200177/DE_371/experiments_WP1/Grand_Ensemble/Final/Pack/')
    
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, 
                        default ='/project/scratch/p200177/DE_371/victorsanchez/results/member_distance/875_member_ensemble/')
    
    parser.add_argument('--gen_sample_dir',type = str, default ="/project/home/p200177/DE_371/experiments_WP1/Grand_Ensemble/Final/Gen/")

    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
    parser.add_argument("--Shape", type=tuple, default=(3,256,256), help='size of the samples')
    parser.add_argument("--crop_indices", type=int, nargs='+', default=[0,256,0,256])
    
     # VGG
    parser.add_argument("--resize_vgg_input", type=float, default=1.0, help="resize input for vgg loss")
    parser.add_argument("--vgg_computation", type=str, default='sol2', choices = ['sol1', 'sol2', 'sol3', 'sol4', 'sol5'], 
                        help="Either we compute layer by layer and member per member but we have to triple th einput to make it rgb or all in one (naive)")
    parser.add_argument("--vgg_state_dict_path", type=str, default='/project/home/p200177/DE_371/resources/vgg_weights/vgg16-random.pth', help="Insert a path")
    parser.add_argument("--vgg_style_layers", type=utils.str2intlist, default=[], help="style layers to include in vgg loss computation")
    parser.add_argument("--vgg_feature_layers", type=utils.str2intlist, default=[0,1,2,3], help="feature layers to include in vgg computation")
    parser.add_argument("--vgg_alpha_feature", type=float, default=1.0, help="weight of the feature/content loss")
    parser.add_argument("--vgg_alpha_style", type=float, default=0.01, help="weight of the style loss")
    parser.add_argument("--vgg_loss_after_step", type=float, default=0, help="compute the vgg loss only after a given number of steps")


    # Dataset information
    parser.add_argument("--normalization", type=str, default="meanmax", choices=["minmax", "meanmax"])
    parser.add_argument('--max_file', type=str, default='MaxNew_4_var.npy') # use 'MaxNew_4_var.npy' if AROME data # max_rr_log.npy
    parser.add_argument('--mean_file', type=str, default='Mean_4_var.npy') # not used if minmax normalization
    parser.add_argument('--min_file', type=str, default='min_rr_log.npy')  # not used if meanmax normalization
    
    parser.add_argument('--device', type=str, default='cuda')

    ########################## CONTROL of Data to invert ######################
    parser.add_argument("--leadtimes", type=utils.str2intlist, default=[6,12,18,24,30,36,42])
    
    parser.add_argument('--start_member', type=int, default=0)
    parser.add_argument('--stop_member', type=int, default=874)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--g_channels', type=int, default=3)
    parser.add_argument('--channel_multiplier', type=int, default=2)

    params = parser.parse_args()


    # fix some of the inputs
    params.Shape = tuple(params.Shape)
    params.crop_indices = tuple(params.crop_indices)

    # create output and pack directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)


    # set the seed for reproduciibility of runs
    seed = params.seed
    torch.manual_seed(seed)

    VGG_loss = PerceptualLoss(config=params,
                              device=params.device
                              ).to(params.device)
    
    ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3)
    # Distance between two members look into : https://climpred.readthedocs.io/en/stable/metrics.html
    # https://confluence.ecmwf.int/display/FUG/Section+6.2.2+Anomaly+Correlation+Coefficient
    #################### main loop ##################
    start = params.start_member
    stop = params.stop_member
    
    for lt in params.leadtimes:
        path_to_pack_sample = params.pack_dir + f"Rsemble_{lt}_875.npy"    
        Ens_r = torch.from_numpy(np.load(path_to_pack_sample)).to(params.device)[:params.stop_member]
        print('shape of AROME ensemble :', Ens_r.size())
        params.date_index = f'{start}_{stop}'
        params.lt_index = lt
        nb_member, _, _ ,_ = Ens_r.size()

        try :
            path_to_sample = params.gen_sample_dir + f"genFsemble_{lt}_875.npy"    
            Ens_gen = torch.from_numpy(shuffle_along_axis(np.load(path_to_sample), 0)).to(params.device)[:params.stop_member]
            nb_gen_member, _, _ ,_ = Ens_gen.shape
        except :
            print(f"File 'genFsemble_{lt}_875.npy' Not Found")
            nb_gen_member=0
        

        dist_l2 = np.zeros((nb_member+nb_gen_member, nb_member+nb_gen_member))
        dist_l1 = np.zeros((nb_member+nb_gen_member, nb_member+nb_gen_member))
        dist_vgg = np.zeros((nb_member+nb_gen_member, nb_member+nb_gen_member))
        dist_ssim = np.zeros((nb_member+nb_gen_member, nb_member+nb_gen_member))
        pearson = np.ones((nb_member+nb_gen_member, nb_member+nb_gen_member))
        anomaly_correlation_coefficient = np.ones((nb_member+nb_gen_member, nb_member+nb_gen_member))
        
        for i in range(nb_member+nb_gen_member):
            for j in range(i, nb_member+nb_gen_member):
                if i!=j :
                    if i < nb_member and j < nb_member :
                        # Distance of AROME members with themselves

                        # MSE
                        dist_l2[i][j] = F.mse_loss(Ens_r[i], Ens_r[j]).cpu()
                        dist_l2[j][i] = dist_l2[i][j]
                        # MAE
                        dist_l1[i][j] = F.l1_loss(Ens_r[i], Ens_r[j]).cpu()
                        dist_l1[j][i] = dist_l1[i][j]
                        # VGG
                        dist_vgg[i][j] = VGG_loss(
                            (Ens_r[i]+1)/2,
                            (Ens_r[j]+1)/2,
                            feature_layers = params.vgg_feature_layers,
                            style_layers = params.vgg_style_layers,
                            alpha_feature = params.vgg_alpha_feature,
                            alpha_style = params.vgg_alpha_style
                        ).cpu()
                        dist_vgg[j][i]=dist_vgg[i][j]
                        # SSIM
                        dist_ssim[i][j] = 1 - ms_ssim_module((Ens_r[i].unsqueeze(0)+1)/2, (Ens_r[j].unsqueeze(0)+1)/2)
                        dist_ssim[j][i]=dist_ssim[i][j]
                        
                        #Pearson correlation coef
                        pearson[i][j]=scipy.stats.pearsonr((Ens_r[i].flatten().cpu()+1)/2, (Ens_r[j].flatten().cpu()+1)/2).statistic
                        pearson[j][i]=pearson[i][j]

                        #ACC
                        anomaly_correlation_coefficient[i][j]=calc_anomaly_correlation_coefficient((Ens_r[i].flatten().cpu()+1)/2, (Ens_r[j].flatten().cpu()+1)/2)
                        # anomaly_correlation_coefficient[i][j]=climpred.metrics._pearson_r((Ens_r[i].flatten().cpu().numpy()+1)/2, (Ens_r[j].flatten().cpu().numpy()+1)/2)
                        anomaly_correlation_coefficient[j][i]=anomaly_correlation_coefficient[i][j]

                    elif i < nb_member and j >=  nb_member :
                        # Distance between AROME and Generated samples

                        # MSE
                        dist_l2[i][j] = F.mse_loss(Ens_r[i], Ens_gen[j-nb_member]).cpu()
                        dist_l2[j][i] = dist_l2[i][j]
                        # MAE
                        dist_l1[i][j] = F.l1_loss(Ens_r[i], Ens_gen[j-nb_member]).cpu()
                        dist_l1[j][i] = dist_l1[i][j]
                        # VGG
                        dist_vgg[i][j] = VGG_loss(
                            (Ens_r[i]+1)/2,
                            (Ens_gen[j-nb_member]+1)/2,
                            feature_layers = params.vgg_feature_layers,
                            style_layers = params.vgg_style_layers,
                            alpha_feature = params.vgg_alpha_feature,
                            alpha_style = params.vgg_alpha_style
                        ).cpu()
                        dist_vgg[j][i]=dist_vgg[i][j]
                        # SSIM
                        dist_ssim[i][j] = 1 - ms_ssim_module((Ens_r[i].unsqueeze(0)+1)/2, (Ens_gen[j-nb_member].unsqueeze(0)+1)/2)
                        dist_ssim[j][i]=dist_ssim[i][j]
                        #Pearson correlation coef
                        pearson[i][j]=scipy.stats.pearsonr((Ens_r[i].flatten().cpu()+1)/2, (Ens_gen[j-nb_member].flatten().cpu()+1)/2).statistic
                        pearson[j][i]=pearson[i][j]
                        #ACC
                        anomaly_correlation_coefficient[i][j]=calc_anomaly_correlation_coefficient((Ens_r[i].flatten().cpu()+1)/2, (Ens_gen[j-nb_member].flatten().cpu()+1)/2)
                        # # anomaly_correlation_coefficient[i][j]=climpred.metrics._pearson_r((Ens_r[i].flatten().cpu().numpy()+1)/2, (Ens_gen[j-nb_member].flatten().cpu().numpy()+1)/2)
                        anomaly_correlation_coefficient[j][i]=anomaly_correlation_coefficient[i][j]

                    elif i >= nb_member and j >= nb_member :
                        # Distance of Generated members with themselves

                        # MSE
                        dist_l2[i][j] = F.mse_loss(Ens_gen[i-nb_member], Ens_gen[j-nb_member]).cpu()
                        dist_l2[j][i] = dist_l2[i][j]
                        # MAE
                        dist_l1[i][j] = F.l1_loss(Ens_gen[i-nb_member], Ens_gen[j-nb_member]).cpu()
                        dist_l1[j][i] = dist_l1[i][j]
                        # VGG
                        dist_vgg[i][j] = VGG_loss(
                            (Ens_gen[i-nb_member]+1)/2,
                            (Ens_gen[j-nb_member]+1)/2,
                            feature_layers = params.vgg_feature_layers,
                            style_layers = params.vgg_style_layers,
                            alpha_feature = params.vgg_alpha_feature,
                            alpha_style = params.vgg_alpha_style
                        ).cpu()
                        dist_vgg[j][i]=dist_vgg[i][j]
                        # SSIM
                        dist_ssim[i][j] = 1 - ms_ssim_module((Ens_gen[i-nb_member].unsqueeze(0)+1)/2, (Ens_gen[j-nb_member].unsqueeze(0)+1)/2)
                        dist_ssim[j][i]=dist_ssim[i][j]
                        #Pearson correlation coef
                        pearson[i][j]=scipy.stats.pearsonr((Ens_gen[i-nb_member].flatten().cpu()+1)/2, (Ens_gen[j-nb_member].flatten().cpu()+1)/2).statistic
                        pearson[j][i]=pearson[i][j]
                        #ACC
                        anomaly_correlation_coefficient[i][j]=calc_anomaly_correlation_coefficient((Ens_gen[i-nb_member].flatten().cpu()+1)/2, (Ens_gen[j-nb_member].flatten().cpu()+1)/2)
                        # # anomaly_correlation_coefficient[i][j]=climpred.metrics._pearson_r((Ens_gen[j-nb_member].flatten().cpu().numpy()+1)/2, (Ens_gen[j-nb_member].flatten().cpu().numpy()+1)/2)
                        anomaly_correlation_coefficient[j][i]=anomaly_correlation_coefficient[i][j]

        dist_l2 = (dist_l2-dist_l2.min())/(dist_l2.max()-dist_l2.min())
        dist_l1 = (dist_l1-dist_l1.min())/(dist_l1.max()-dist_l1.min())
        dist_vgg = (dist_vgg-dist_vgg.min())/(dist_vgg.max()-dist_vgg.min())
        dist_ssim = (dist_ssim-dist_ssim.min())/(dist_ssim.max()-dist_ssim.min())

        fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(25,6))
        im=ax[0].matshow(dist_l2, cmap='plasma')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title('L2 distance')
        fig.colorbar(im, ax=ax[0], shrink=0.5)
        im=ax[1].matshow(dist_l1, cmap='plasma')
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title('L1 distance')
        fig.colorbar(im, ax=ax[1], shrink=0.5)
        im=ax[2].matshow(dist_vgg, cmap='plasma')
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[2].set_title('VGG distance')
        fig.colorbar(im, ax=ax[2], shrink=0.5)
        im=ax[3].matshow(dist_ssim, cmap='plasma')
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        ax[3].set_title('MS-SSIM distance')
        fig.colorbar(im, ax=ax[3], shrink=0.5)
        im=ax[4].matshow(pearson, cmap='plasma')
        ax[4].set_xticks([])
        ax[4].set_yticks([])
        ax[4].set_title('Pearson Correlation')
        fig.colorbar(im, ax=ax[4], shrink=0.5)
        im=ax[5].matshow(anomaly_correlation_coefficient, cmap='plasma')
        ax[5].set_xticks([])
        ax[5].set_yticks([])
        ax[5].set_title('Anomaly Correlation Coefficient')
        fig.colorbar(im, ax=ax[5], shrink=0.5)

        fig.tight_layout()
        fig.savefig(params.output_dir+f'distance_between_member_{lt}_{nb_gen_member}.png')


"""
This script compares the distance between ensemble forecast members with different distance criterions
"""
import torch
import argparse
import os
import numpy as np
import yaml
import pandas as pd
import torch.nn.functional as F
print('Importing Generator')
print('Importing perturbation utils')
import perturbation.utils as utils
import matplotlib.pyplot as plt
from inversion.vgg_perceptual_loss import VGGPerceptualLoss
from inversion.ssim import MS_SSIM
import scipy
torch.manual_seed(42) #reproducibility of runs

def calc_anomaly_correlation_coefficient(x,y):
    r"""Pearson product-moment correlation coefficient.

    A measure of the linear association between the forecast and verification data that
    is independent of the mean and variance of the individual distributions. This is
    also known as the Anomaly Correlation Coefficient (ACC) when correlating anomalies.

    .. math::
        corr = \frac{cov(f, o)}{\sigma_{f}\cdot\sigma_{o}},

    where :math:`\sigma_{f}` and :math:`\sigma_{o}` represent the standard deviation
    of the forecast and verification data over the experimental period, respectively.
    
    Args:

    """
    return torch.cov(torch.cat((x.unsqueeze(1),y.unsqueeze(1)), dim=1)) / (torch.std(x,dim=0, unbiased=True)*torch.std(y,dim=0, unbiased=True))
    

if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################
    # Checkpoint directory - PATH to generator's weight
    parser.add_argument('--ckpt_dir', type = str, 
                        default ='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt')
    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str, 
                        default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, 
                        default ='/project/scratch/p200177/DE_371/victorsanchez/results/member_distance/')
    parser.add_argument('--gen_sample_dir',type = str, default ="/project/home/p200177/DE_371/experiments_WP1/inversion_process_analysis/final_inversion_on_test_set/perceptual_exp45/perturbation/stochastic_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_False_-1_16_/samples/")

    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
    parser.add_argument("--Shape", type=tuple, default=(3,256,256), help='size of the samples')
    parser.add_argument("--crop_indices", type=int, nargs='+', default=[0,256,0,256])
    
     # VGG
    parser.add_argument("--resize_vgg_input", type=float, default=1.0, help="resize input for vgg loss")
    parser.add_argument("--vgg_computation", type=str, default='sol2', choices = ['sol1', 'sol2', 'sol3', 'sol4', 'sol5'], 
                        help="Either we compute layer by layer and member per member but we have to triple th einput to make it rgb or all in one (naive)")
    parser.add_argument("--vgg_state_dict_path", type=str, default='/project/scratch/p200177/DE_371/resources/vgg_weights/vgg16-random.pth', help="Insert a path")
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
    parser.add_argument("--dates_file", type=str, default = 'Large_lt_test_labels.csv')
    parser.add_argument("--date_start", type=str, default = "2021-10-02")
    parser.add_argument("--date_stop", type=str, default = "2021-10-03")
    parser.add_argument("--leadtimes", type=utils.str2intlist, default=[3,6,9,12,15,18,21,24,27,30,33,36,39,42,44])
    
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

    ################## loading dates and file names ##
    df = pd.read_csv(params.real_data_dir + params.dates_file)
    df_date = df.copy()
    df_date['Date'] = pd.to_datetime(df_date['Date'])
    df_extract = df_date[(df_date['Date']>=params.date_start) & (df_date['Date']<=params.date_stop)]
    
    list_dates = df_extract['Date'].unique()
    Means=None
    Maxs=None
    Mins=None
    if params.normalization=="meanmax":
        Means = np.load(f'{params.real_data_dir}{params.mean_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
        Maxs = np.load(f'{params.real_data_dir}{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    elif params.normalization=="minmax":
       Mins = np.load(f'{params.real_data_dir}/stat_files/{params.min_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
       Maxs = np.load(f'{params.real_data_dir}/stat_files/{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    else:
       raise ValueError(f"Unknown normalization: {params.normalization}")

    VGG_loss = VGGPerceptualLoss(
                            state_dict_path=params.vgg_state_dict_path,
                            init_layer=True if params.vgg_computation=='sol4' else False,
                            vgg_single_channel_input=True if params.vgg_computation=='sol5' else False
            ).to(params.device)
    
    ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3)
    # Distance between two members look into : https://climpred.readthedocs.io/en/stable/metrics.html
    # https://confluence.ecmwf.int/display/FUG/Section+6.2.2+Anomaly+Correlation+Coefficient
    #################### main loop ##################
    
    for date_ in list_dates:
        print(date_)
        datename = date_.strftime('%Y-%m-%d')
        print("\n===========================")
        for lt in params.leadtimes:
            params.date_index = datename
            params.lt_index = lt

            print('Launching inversion process for the date {} with leadtime {}.'.format(datename,lt))
            df0 = df_extract[(df_extract['Date']==date_) & (df_extract['LeadTime']==lt-1)]
            if len(df0)==0:
                print("# samples: 0")
                continue
            Ens_r = utils.load_batch_from_timestamp(
                df_extract, 
                date_, 
                lt-1, 
                params.real_data_dir, 
                Shape=params.Shape, 
                var_indices=params.var_indices,
                normalization=params.normalization,
                Means=Means,
                Mins=Mins,
                Maxs=Maxs
                
            ).to(params.device)

            try :
                path_to_sample = params.gen_sample_dir + f"genFsemble_{datename}_{lt}_1000_16_generated_pert.npy"    
                Ens_gen = torch.from_numpy(np.load(path_to_sample)).to(params.device)
            except :
                print(f"File 'genFsemble_{date_}_{lt}_1000_16_generated_pert.npy' Not Found")

            nb_member, _, _ ,_ = Ens_r.shape
            nb_gen_member, _, _ ,_ = Ens_gen.shape

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
                            # anomaly_correlation_coefficient[i][j]=calc_anomaly_correlation_coefficient((Ens_r[i].flatten().cpu()+1)/2, (Ens_r[j].flatten().cpu()+1)/2)
                            # anomaly_correlation_coefficient[i][j]=climpred.metrics._pearson_r((Ens_r[i].flatten().cpu().numpy()+1)/2, (Ens_r[j].flatten().cpu().numpy()+1)/2)
                            # anomaly_correlation_coefficient[j][i]=anomaly_correlation_coefficient[i][j]

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
                            # anomaly_correlation_coefficient[i][j]=calc_anomaly_correlation_coefficient((Ens_r[i].flatten().cpu()+1)/2, (Ens_gen[j-nb_member].flatten().cpu()+1)/2)
                            # # anomaly_correlation_coefficient[i][j]=climpred.metrics._pearson_r((Ens_r[i].flatten().cpu().numpy()+1)/2, (Ens_gen[j-nb_member].flatten().cpu().numpy()+1)/2)
                            # # anomaly_correlation_coefficient[j][i]=anomaly_correlation_coefficient[i][j]

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
                            # anomaly_correlation_coefficient[i][j]=calc_anomaly_correlation_coefficient((Ens_gen[i-nb_member].flatten().cpu()+1)/2, (Ens_gen[j-nb_member].flatten().cpu()+1)/2)
                            # # anomaly_correlation_coefficient[i][j]=climpred.metrics._pearson_r((Ens_gen[j-nb_member].flatten().cpu().numpy()+1)/2, (Ens_gen[j-nb_member].flatten().cpu().numpy()+1)/2)
                            # # anomaly_correlation_coefficient[j][i]=anomaly_correlation_coefficient[i][j]

            dist_l2 = (dist_l2-dist_l2.min())/(dist_l2.max()-dist_l2.min())
            dist_l1 = (dist_l1-dist_l1.min())/(dist_l1.max()-dist_l1.min())
            dist_vgg = (dist_vgg-dist_vgg.min())/(dist_vgg.max()-dist_vgg.min())
            dist_ssim = (dist_ssim-dist_ssim.min())/(dist_ssim.max()-dist_ssim.min())

            fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(16,5))
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
            # im=ax[5].matshow(anomaly_correlation_coefficient, cmap='plasma')
            # ax[5].set_xticks([])
            # ax[5].set_yticks([])
            # ax[5].set_title('Anomaly Correlation Coefficient')
            # fig.colorbar(im, ax=ax[5], shrink=0.5)

            fig.tight_layout()
            fig.savefig(params.output_dir+f'distance_between_member_{datename}_{lt}_{nb_gen_member}.png')


"""
This script compares the distance between ensemble forecast members with different distance criterions
"""
import torch
import argparse
import os
import numpy as np
import torch.nn.functional as F
import utils.utils as utils
import matplotlib.pyplot as plt
from inversion.experimental_loss.ssim import MS_SSIM
from plot_analysis.utils import calc_anomaly_correlation_coefficient
import scipy
from utils.utils import denormalize, normalize
torch.manual_seed(42) #reproducibility of runs


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################
    # Checkpoint directory - PATH to generator's weight
    parser.add_argument('--ckpt_dir', type = str, default ='/project/home/p200177/DE_371/resources/models/trained_generator/000024.pt')
    # Real Data Directory - PATH to samples of the dataset
    
    parser.add_argument('--pack_dir', type = str, default ='/project/home/p200177/DE_371/experiments_WP1/Grand_Ensemble/Pack/')
    parser.add_argument('--real_data_dir', type = str, 
                        default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, default ='/project/home/p200177/DE_371/experiments_WP1/Grand_Ensemble/Scores/Optim/Distance_plots/')
    
    parser.add_argument('--gen_data_dir',type = str, default ="/project/home/p200177/DE_371/experiments_WP1/Grand_Ensemble/Perturbation/Optim/stochastic_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_False/samples/")

    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
    parser.add_argument("--Shape", type=tuple, default=(3,256,256), help='size of the samples')
    parser.add_argument("--crop_indices", type=int, nargs='+', default=[0,256,0,256])
    
    # Dataset information
    parser.add_argument("--normalization", type=str, default="meanmax", choices=["minmax", "meanmax"])
    parser.add_argument('--max_file', type=str, default='MaxNew_4_var.npy') # use 'MaxNew_4_var.npy' if AROME data # max_rr_log.npy
    parser.add_argument('--mean_file', type=str, default='Mean_4_var.npy') # not used if minmax normalization
    parser.add_argument('--min_file', type=str, default='min_rr_log.npy')  # not used if meanmax normalization
    
    parser.add_argument('--device', type=str, default='cpu')

    ########################## CONTROL of Data to invert ######################
    parser.add_argument("--leadtimes", type=utils.str2intlist, default=[6,12,18,24,30,36,42])
    
    parser.add_argument('--inv_step', type=int, default=2000)
    parser.add_argument('--start_member', type=int, default=0)
    parser.add_argument('--stop_member', type=int, default=874)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--g_channels', type=int, default=3)
    parser.add_argument('--channel_multiplier', type=int, default=2)

    params = parser.parse_args()


    # fix some of the inputs
    params.Shape = tuple(params.Shape)
    params.crop_indices = tuple(params.crop_indices)

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
    
    # create output and pack directories
    os.makedirs(params.output_dir, exist_ok=True)


    # set the seed for reproduciibility of runs
    seed = params.seed
    torch.manual_seed(seed)

    
    ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3)
    # Distance between two members look into : https://climpred.readthedocs.io/en/stable/metrics.html
    # https://confluence.ecmwf.int/display/FUG/Section+6.2.2+Anomaly+Correlation+Coefficient
    #################### main loop ##################
    start = params.start_member
    stop = params.stop_member
    random = False
    if random : 
        members_id = np.random.randint(0, 875, params.stop_member).tolist()

    for lt in params.leadtimes:
        path_to_pack_sample = params.pack_dir + f"Rsemble_{lt}_875.npy"
        print(f'Start import of AROME Ensemble of leadtime {lt}')
        
        ens_r=utils.collate_R_ensemble(
            data_dir=params.pack_dir,
            members=list(range(start, stop)),
            lead_time=lt,
            var_indices=params.var_indices
        )
        
        if not random :
            Ens_r = torch.from_numpy(ens_r).to(params.device)[:params.stop_member]
        else :
            Ens_r = torch.from_numpy(ens_r).to(params.device)[members_id]
        
        
        print('Import done. Shape of AROME ensemble :', Ens_r.size())
        params.date_index = f'{start}_{stop}'
        params.lt_index = lt
        nb_member, _, _ ,_ = Ens_r.size()

        # try :
        path_to_sample = params.gen_data_dir + f"genFsemble_{lt}_875.npy"   
        print(f'Start import of Generated Ensemble of leadtime {lt}') 
        gen_ens = utils.collate_gen_ensemble(
            data_dir=params.gen_data_dir,
            members=list(range(start, stop)),
            lead_time=lt,
            var_indices=params.var_indices,
            inv_step=params.inv_step
        )

        # Ens_gen = torch.from_numpy(shuffle_along_axis(np.load(path_to_sample), 0)).to(params.device)[:params.stop_member]
        if not random :
            Ens_gen = torch.from_numpy(gen_ens).to(params.device)[:params.stop_member]
        else :
            Ens_gen = torch.from_numpy(gen_ens).to(params.device)[members_id]

        print('Import done. Shape of Generated ensemble :', Ens_gen.size())

        Ens_gen = normalize(
            data=Ens_gen.clone(),
            normalization_type=params.normalization,
            Means=Means,
            Mins=Mins,
            Maxs=Maxs,
            apply_log_transform=False
        )

        nb_gen_member, _, _ ,_ = Ens_gen.shape
        # except :
        #     print(f"File 'genFsemble_{lt}_875.npy' Not Found")
        #     nb_gen_member=0
        
        print('Total nb member to compare :', nb_member+nb_gen_member)
        dist_l2 = np.zeros((nb_member+nb_gen_member, nb_member+nb_gen_member))
        dist_l1 = np.zeros((nb_member+nb_gen_member, nb_member+nb_gen_member))
        dist_ssim = np.zeros((nb_member+nb_gen_member, nb_member+nb_gen_member))
        pearson = np.ones((nb_member+nb_gen_member, nb_member+nb_gen_member))
        anomaly_correlation_coefficient = np.ones((nb_member+nb_gen_member, nb_member+nb_gen_member))
        
        
        for i in range(nb_member+nb_gen_member):
            for j in range(i, nb_member+nb_gen_member):
                # print(lt, i,j)
                if i!=j :
                    if i < nb_member and j < nb_member :
                        # Distance of AROME members with themselves

                        # MSE
                        dist_l2[i][j] = F.mse_loss(Ens_r[i], Ens_r[j]).cpu()
                        dist_l2[j][i] = dist_l2[i][j]
                        # MAE
                        dist_l1[i][j] = F.l1_loss(Ens_r[i], Ens_r[j]).cpu()
                        dist_l1[j][i] = dist_l1[i][j]
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

        # dist_l2 = (dist_l2-dist_l2.min())/(dist_l2.max()-dist_l2.min())
        # dist_l1 = (dist_l1-dist_l1.min())/(dist_l1.max()-dist_l1.min())
        # dist_ssim = (dist_ssim-dist_ssim.min())/(dist_ssim.max()-dist_ssim.min())
        output_dir_data = params.output_dir+'data/'
        output_dir_plot = params.output_dir+'plots/'

        os.makedirs(output_dir_data, exist_ok=True)
        os.makedirs(output_dir_plot, exist_ok=True)

        np.save(output_dir_data+f'dist_l2_between_member_{lt}_{nb_gen_member}.npy', dist_l2)
        np.save(output_dir_data+f'dist_l1_between_member_{lt}_{nb_gen_member}.npy', dist_l1)
        np.save(output_dir_data+f'dist_lssim_between_member_{lt}_{nb_gen_member}.npy', dist_ssim)
        np.save(output_dir_data+f'pearson_between_member_{lt}_{nb_gen_member}.npy', pearson)
        np.save(output_dir_data+f'acc_between_member_{lt}_{nb_gen_member}.npy', anomaly_correlation_coefficient)


        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(25,6))
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
        im=ax[2].matshow(dist_ssim, cmap='plasma')
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[2].set_title('MS-SSIM distance')
        fig.colorbar(im, ax=ax[2], shrink=0.5)
        im=ax[3].matshow(pearson, cmap='plasma')
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        ax[3].set_title('Pearson Correlation')
        fig.colorbar(im, ax=ax[3], shrink=0.5)
        im=ax[4].matshow(anomaly_correlation_coefficient, cmap='plasma')
        ax[4].set_xticks([])
        ax[4].set_yticks([])
        ax[4].set_title('Anomaly Correlation Coefficient')
        fig.colorbar(im, ax=ax[4], shrink=0.5)
        fig.tight_layout()
        fig.savefig(output_dir_plot+f'distance_between_member_{lt}_{nb_gen_member}.png')

        
        
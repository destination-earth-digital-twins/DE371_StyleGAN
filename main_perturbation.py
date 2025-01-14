#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:21:37 2023

@author: brochetc

Main pod sampling script

"""

import torch
import argparse
import os
import numpy as np
import pickle
import pandas as pd
from collections import OrderedDict
from gan.model.stylegan2 import Generator
from ast import literal_eval as make_tuple
import utils.utils as utils
import perturbation.smpca as smpca
from shutil import copyfile
from inversion.plotter import online_pert_plot, online_pert_diff_plot


def str2list(li):
    if type(li)==list:
        li2 = li
        return li2
    
    elif type(li)==str:
        li2=li[1:-1].split(',')
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))
        

def compute_generate_save(G, params, metrics_list, Means, Mins, Maxs, apply_log_transform):

    N_samples = params.N_samples
    if params.verbose:
        print(datename, lt)
        print(params.date_index, params.lt_index)

    Ens_r = torch.tensor(np.load(params.pack_dir+f'Rsemble_{datename}_{lt}.npy'), dtype = torch.float32)
    w_ens = torch.tensor(np.load(params.data_dir + f'w_{params.date_index}_{params.lt_index}_{params.inv_step}.npy').astype(np.float32))
    inv_ens=np.load(params.data_dir + f'invertFsemble_{params.date_index}_{params.lt_index}_{params.inv_step}.npy').astype(np.float32)
    Ens_feature=None
    if params.feature_insertion:
        Ens_feature=torch.tensor(np.load(params.data_dir + f'feature_{params.date_index}_{params.lt_index}_{params.inv_step}.npy').astype(np.float32))
    # subsampling if N_conditioners is lower than initial ensemble size
    if (params.N_conditioners<w_ens.shape[0]):
        cond_indices = np.random.choice(range(w_ens.shape[0]), params.N_conditioners)
        w_ens = w_ens[cond_indices]
        Ens_r = Ens_r[cond_indices]
    print('############### Perturbating ###############')
    print('loading generation hyperparams')
    Whitening = torch.load(params.eigendir + 'Whitening.pt') if params.sample_rule=='stochastic' else None
    Coloring = torch.load(params.eigendir + 'Coloring.pt') if params.sample_rule=='stochastic' else None
    w0 = torch.load(params.eigendir + 'latent_mean.pt') if params.sample_rule=='stochastic' else None
    betas = torch.tensor(np.load(os.path.join(params.scale_dir,"ema_scale.npy")).astype(np.float32)[params.scale_interp_step], device=params.device)
    alphas = torch.tensor(np.load(os.path.join(params.scale_dir,"ema_interp.npy")).astype(np.float32)[params.scale_interp_step], device=params.device)

    if params.import_perturbation:
        print(f'Importing perturbation from {params.path_perturbation}')
        
    gen, w_new = smpca.sm_pca(
        Ens_w=w_ens, 
        G=G, 
        N_samples=N_samples, 
        sm_ind=params.style_indices,
        device=params.device, 
        sample_rule=params.sample_rule, 
        betas=betas,
        alphas=alphas,
        verbose=params.verbose,
        Whitening=Whitening,
        Coloring=Coloring,
        w0=w0,
        import_perturbation=params.import_perturbation,
        save_perturbation=params.save_perturbation,
        path_perturbation=params.path_perturbation,
        Ens_feature=Ens_feature,
        feature_id=params.feature_id,
        feature_scale=params.feature_scale,
        temporal_consistency=False, #TODO : Add this to config
        w_pert_former=None, #TODO : Add this to config
        rho=0.1, #TODO : Add this to config
        dt=1 #TODO : Add this to config
    )

    if params.verbose:
        print(gen.mean(axis=(0,-2,-1)))
        print(Ens_r.mean(axis=(0,-2,-1)))
    
    title = f'{params.date_index}_{params.lt_index}_{params.inv_step}_{params.N_conditioners}'
    
    
    if params.save_w_perturbated:
        if params.save_perturbation and not params.import_perturbation :
            title+='_generated_pert'
            np.save(params.output_dir + f'/samples/w_pert_{title}.npy', w_new[0])
        elif params.import_perturbation:
            title+='_imported_pert'
            np.save(params.output_dir + f'/samples/w_pert_{title}.npy', w_new)
        else :
            title+='_generated_pert'
            np.save(params.output_dir + f'/samples/w_pert_{title}.npy', w_new)
    if params.save_perturbation and not params.import_perturbation :
        np.save(params.output_dir + f'/samples/perturbation_{title}.npy', w_new[1])

    if params.import_normalized_data:
        Ens_r_denorm = utils.denormalize(
                                        data=Ens_r,
                                        normalization_type=params.normalization,
                                        Means=Means,
                                        Mins=Mins,
                                        Maxs=Maxs,
                                        apply_log_transform=apply_log_transform
                                        )
        inv_ens_denorm = utils.denormalize(
                                        data=inv_ens,
                                        normalization_type=params.normalization,
                                        Means=Means,
                                        Mins=Mins,
                                        Maxs=Maxs,
                                        apply_log_transform=apply_log_transform
                                        )
    gen_denorm = utils.denormalize(
                                    data=gen,
                                    normalization_type=params.normalization,
                                    Means=Means,
                                    Mins=Mins,
                                    Maxs=Maxs,
                                    apply_log_transform=apply_log_transform
                                    )
    
    if params.save_normalized_sample:
        np.save(params.output_dir + f'/samples/genFsemble_{title}.npy', gen)
    else:
        np.save(params.output_dir + f'/samples/genFsemble_{title}.npy', gen_denorm)
    

    online_pert_plot(
        packsample=Ens_r_denorm.numpy(), 
        invsample=inv_ens_denorm, 
        pert_sample=gen_denorm,
        crop=[0,-1,0,-1],
        mem_idx=0 if params.N_conditioners>1 else cond_indices, 
        figtitle=f"Generated samples for {title}", 
        figname=params.output_dir + f"/samples/genFsemble_{title}.png"
    )

    online_pert_diff_plot(
        invsample=inv_ens_denorm, 
        pert_sample=gen_denorm,
        crop=[0,-1,0,-1],
        mem_idx=0 if params.N_conditioners>1 else cond_indices, 
        figtitle=f"Generated samples for {title}", 
        figname=params.output_dir + f"/samples/diff_genFsemble_{title}.png"
    )

    if params.runtime_metrics:
        dic = {'Mean' : {'real':Ens_r_denorm.mean(axis=(0,-2,-1)), 'fake':gen_denorm.mean(axis=(0,-2,-1))},
             'Std' : {'real': np.sqrt(Ens_r_denorm.var(axis=0).mean(axis=(-2,-1))), 'fake': np.sqrt(gen_denorm.var(axis=0).mean(axis=(-2,-1)))},
             'Max' : {'real':Ens_r_denorm.max(axis=(0,-2,-1)), 'fake':gen_denorm.max(axis=(0,-2,-1))},
             'Align' : 1.0 - (gen_denorm.std(axis=0) * Ens_r_denorm.std(axis=0)).sum(axis=(-2,-1)) / (np.sqrt((Ens_r_denorm.std(axis=0) ** 2).sum(axis=(-2,-1))) *  np.sqrt((gen0.std(axis=0) **2).sum(axis=(-2,-1)))),
        }
        if params.verbose: print(dic)
        return dic
    return {}
     
if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################
    # Checkpoint directory - PATH to generator's weight
    parser.add_argument('--ckpt_dir', type = str, default ='')
    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str, default='')
    # Data Directory - PATH to samples from inversion process                    
    parser.add_argument('--data_dir', type=str, default='')
    # Pack Directory - PATH where the packed ensembles will be saved
    parser.add_argument("--pack_dir", type=str, default = '')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, default ='')
    parser.add_argument('--path_root_readme',type = str, default ='')
    parser.add_argument("--import_normalized_data", action="store_true",
                        help='Flag to remark that imported data are normalized. If True, data will be denormalized !')

    # Generator network information
    parser.add_argument('--add_name',type = str, default='')
    parser.add_argument('--eigendir',type = str, default ='')
    
    # Feature incorporation
    # See https://arxiv.org/pdf/2202.02183 for more info on Feature insertion
    parser.add_argument('--feature_insertion', action="store_true", help='Whether to insert the predicted features into the generator.')
    parser.add_argument('--feature_scale', type=float, default=1, help='Scale of inserted features')
    parser.add_argument('--feature_id', type=int, default=6, choices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], help='id of feature to insert')
    
    # Dataset information
    parser.add_argument("--normalization", type=str, default="meanmax", choices=["minmax", "meanmax", ""])
    parser.add_argument('--max_file', type=str, default='') # use 'MaxNew_4_var.npy' if AROME data # max_rr_log.npy
    parser.add_argument('--mean_file', type=str, default='') # not used if minmax normalization
    parser.add_argument('--min_file', type=str, default='')  # not used if meanmax normalization
    parser.add_argument('--save_normalized_sample', action='store_true')

    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
    parser.add_argument("--Shape", type=make_tuple, default=(3,256,256), help='size of the samples')
    parser.add_argument("--N_samples", type=int, default=10, help='number of new samples') 
    parser.add_argument("--N_conditioners",type=int, default=16, help="number of 'seed' samples used for conditioning")
    parser.add_argument("--inv_step", type=int, default=2000, help='step of inversion to load w')
    
    
    ######################## PERTURBATION PARAMETERS #######################
    parser.add_argument('--sample_rule', type=str, default='stochastic', 
                        choices = ['stochastic', 'extrapolation'])

    parser.add_argument('--style_indices', type = str2list, default='[1,1,1,1,1,1,1,1,1,1,0,0,0,0]')
    parser.add_argument('--unbias', action="store_true")

    parser.add_argument('--scale_dir', type=str, default="")
    parser.add_argument('--scale_interp_step',type=int, default=-1)
    # w_perturbated = w_inv + perturbation
    parser.add_argument("--save_w_perturbated", action="store_true", help='Save final perturbated latent space')
    parser.add_argument("--import_perturbation", action="store_true", help='Flag to import perturbation')
    parser.add_argument("--save_perturbation", action="store_true", help='Flag to save perturbation')
    parser.add_argument('--path_perturbation',type=str, default ="", help='Flag perturbation path')

    ########################## CONTROL of Data to perturb ######################
    parser.add_argument("--dates_file", type=str, default = 'Large_lt_test_labels.csv')
    parser.add_argument("--date_start", type=str, default = "2021-07-01")
    parser.add_argument("--date_stop", type=str, default = "2021-07-02")
    parser.add_argument("--leadtimes", type=utils.str2intlist, default= [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45])

    ###########################################################################
    parser.add_argument("--runtime_metrics", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument('--device', type=str, default='cuda') # if torch.cuda.is_available() else 'cpu')

    params = parser.parse_args()
    params.output_dir = params.output_dir + f"{params.sample_rule}_{params.style_indices}_{params.unbias}_{params.scale_interp_step}_{params.N_conditioners}_{params.add_name}/" 

    # create output directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)

    ################## selecting dates
    print('reading dates')
    df = pd.read_csv(params.real_data_dir + params.dates_file)
    df_date = df.copy()
    df_date['Date'] = pd.to_datetime(df_date['Date'])
    df_extract = df_date[(df_date['Date']>=params.date_start) & (df_date['Date']<=params.date_stop)]
    liste_dates = df_extract['Date'].unique()
    
    ################## carrying scaling info to pass it whenever needed
    Means=None
    Maxs=None
    Mins=None
    if params.normalization=="meanmax":
        Means = np.load(f'{params.real_data_dir}{params.mean_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
        Maxs = np.load(f'{params.real_data_dir}{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    elif params.normalization=="minmax":
       Mins = np.load(f'{params.real_data_dir}stat_files/{params.min_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
       Maxs = np.load(f'{params.real_data_dir}stat_files/{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
       Means = np.load(f'{params.real_data_dir}stat_files/{params.mean_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    elif params.normalization=="":
        pass
    else:
       raise ValueError(f"Unknown normalization: {params.normalization}")
    
    ############################################################
    if not os.path.exists(params.output_dir + 'samples/'):
        os.mkdir(params.output_dir + 'samples/')
    if not os.path.exists(params.output_dir + 'log/'):
        os.mkdir(params.output_dir + 'log/')
    source_readme = params.path_root_readme
    target_readme = params.output_dir + 'ReadMe_0.txt'
    copyfile(source_readme, target_readme)
    
    
    ################ loading network #################

    
    print('loading G')
    G = Generator(params.Shape[1], 512,n_mlp=8,nb_var=params.Shape[0])
    ckpt = torch.load(params.ckpt_dir, map_location='cpu')['g_ema']
    if 'module' in list(ckpt.items())[0][0]: #juggling with Pytorch versioning and different module packaging
        ckpt_adapt = OrderedDict()
        for k in ckpt.keys():
            k0 = k[7:]
            ckpt_adapt[k0] = ckpt[k]
        G.load_state_dict(ckpt_adapt)
    else:
        G.load_state_dict(ckpt)
    G.eval()
    G = G.to(params.device)

    #############################  Main loop ###############################
    
    metrics_list = ['variance', 'std_diff']#, 'mean_bias']
    metrics = {}
    for date_ in liste_dates:
        datename = date_.strftime('%Y-%m-%d')
        for lt in params.leadtimes:
            params.date_index = datename
            params.lt_index = lt
            
            already_exist = []
            if not params.import_perturbation :
                label = 'generated_pert'
            else :
                label = 'imported_pert'
            if os.path.isfile(params.output_dir + f'/samples/genFsemble_{params.date_index}_{params.lt_index}_{params.inv_step}_{params.N_conditioners}_{label}.npy'):
                already_exist.append(True)
            else :
                already_exist.append(False)
            if os.path.isfile(params.output_dir + f'/samples/w_pert_{params.date_index}_{params.lt_index}_{params.inv_step}_{params.N_conditioners}_{label}.npy'):
                already_exist.append(True)
            else :
                already_exist.append(False)
            if np.all(already_exist) :
                print('The perturbation was already done for the date {} with leadtime {}. This sample is skipped.'.format(datename,lt))
            else :
                print('Launching perturbation process for the date {} with leadtime {}.'.format(datename,lt))    
                try:
                    print('generating')
                    metrics[(datename,lt)] = compute_generate_save(
                        G=G,
                        params=params,
                        metrics_list=metrics_list,
                        Means=Means,
                        Mins=Mins,
                        Maxs=Maxs,
                        apply_log_transform=True if params.Shape[0]==4 else False
                    )
                except FileNotFoundError as e:
                    print(f"File not found {e}")
                    pass
            
    path = params.output_dir + 'log/'
    pickle.dump(metrics,open(path+'metrics.p','wb'))

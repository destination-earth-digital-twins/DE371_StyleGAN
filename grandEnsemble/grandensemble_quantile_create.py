#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import argparse
import os
from tqdm import trange

def str2intlist(li):
    if type(li)==list:
        li2 = [int(p) for p in li]
        return li2
    
    elif type(li)==str:
        li2 = li[1:-1].split(',')
        li3 = [int(p) for p in li2]
        return li3

    else : 
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))

font = {
    "family": "serif",
    "color": "black",
    "weight": "normal",
    "size": 25,
}

def plot_quantile(
    data_list,
    output_dir,
    name_quantile,
    param,
    leadtime,
    clim,
    id_quantile_to_plot = [0,1,6,9,12],
    denom = ["Q0","Q05", "Q1", "Q5", "Q10","Q25","Q50","Q75","Q90","Q95", "Q99", "Q995", "Q100", "Sdev"]
    ):

    fig, axs = plt.subplots(
            nrows=1, ncols=len(id_quantile_to_plot), figsize=(15, 5)
        )
    cmap = plt.get_cmap("PiYG", 7)
    os.makedirs(output_dir+f'/{name_quantile}/', exist_ok=True)
    for idx, id_quantile in enumerate(id_quantile_to_plot):
        name = denom[id_quantile]
        quantile = data_list[id_quantile]

        im = axs[idx].imshow(
            quantile,
            origin="lower",
            cmap=cmap,
            clim=clim[id_quantile]
        )
        fig.colorbar(im, ax=axs[idx], shrink=0.5)
        axs[idx].set_title(F'{name}', fontdict=font)

        fig.suptitle(f"Quantiles of {name_quantile} for {param} variable", fontdict=font)
        fig.tight_layout()
        fig.savefig(
            output_dir+f'/{name_quantile}/plot_{name_quantile}_{param}_{leadtime}.pdf'
        )
        plt.close()

if __name__=="__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument('--param', type=str2intlist, default=['ff', 't2m'])
    parser.add_argument('--base_dir', type=str, default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/grandEnsemble/AROME/')
    parser.add_argument('--GAN_sample_dir', type=str, default='/project/home/p200177/DE_371/experiments_WP1/Grand_Ensemble/Perturbation/')
    parser.add_argument('--output_dir', type=str, default='/project/home/p200177/DE_371/experiments_WP1/Grand_Ensemble/Scores/test_test')
    parser.add_argument('--leadtimes', type=str2intlist, default=[6,12,18,24,30,36,42]) # echeance de la prevision, n'importe quelle valeur entre 0 et 45h est disponible (par pas de 1h)
    parser.add_argument('--inv_step', type=int, default=1000)
    parser.add_argument('--unbias', action="store_true")
    args = parser.parse_args()


    ###SetUp 
    myDates = '2021100121-2021100121-PT1H' # Start-End-Step with YYYYMMDDHH format HH= reseau pour PEARO c'est 03 09 15 ou 21
    suite = "GMGI"                         # On va chercher les fichiers de l'archive oper, suite="G6CN" : on va chercher les fichiers de l'experience de recherche G6CN
    members = list(range(1,2,1))   # les 16 membres PEARO  <-- le "2" pour le 2e run audn AllMB a été chargé     
    lag = 0                      # duree cumul precip
    quant = [0,0.5,1,5,10,25,50,75,90,95,99,99.5,100]
    Nsmall = 16
    Nlbc = 25
    nbGANs = 1       # Nb of different GAN setups to plot
    nbrandinit = 50
    nameGAN = ["stochastic_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_False/"]
    GANfilenames = ['genFsemble_']
    GANnameout = ['Optim_MSE'] 
    plot_id_nbrandinit = 0
    os.makedirs(args.output_dir, exist_ok=True)

    # Domains
    dom = 'GAN'

    # Initialisation projection
    lplotq=True
    lrandinit=False

    def initsmall(lstbc,lstic,Ns,Nlbc):
        r''' Tirage aléatoire des membres conditionneurs '''
        yic = random.sample(lstic, Ns)
        ybc = random.sample(lstbc, Ns)
        mb=np.zeros((Ns))
        # Find members corresponding to (yic,ybc) pairs
        for k in range(Ns):
            loc_bc=np.where(np.asarray(lstbc)==ybc[k])
            #index member of the PEARO experiment start from 1
            #if python storage of members start at 0 remove '+1'
            mb[k]=(yic[k]-1)*Nlbc + loc_bc[0][0]
        return mb

    lstlbc = [2,20,9,5,32,15,19,21,13,1,34,12,10,31,23,11,8,24,29,22,28,25,6,33,14,7,30,27,0,18,4,26,3,16,17]
    lstic  = list(range(1,26))
    for param in args.param:
        for leadtime in args.leadtimes :
            # Tirage nrandinit ensembles conditionneurs
            if lrandinit:
                print("tirage aléatoire des membres conditionneurs")
                mb = np.zeros((Nsmall,nbrandinit))
                for r in range(nbrandinit):
                    mb[:,r] = initsmall(lstlbc,lstic,Nsmall,Nlbc)
                np.save(args.output_dir + '/' + 'nbrandinit_MBs',mb,allow_pickle=True)

            reseau = '2021-10-01T21:00:00Z'
            clim = list()

            ################################################
            ############# Large AROME Ensemble #############
            ################################################

            print("Loading large real ensemble")
            tabref = np.load(args.base_dir + '/AllMB_domain_' + str(dom) + '_' + param + str(lag)  + "_" + suite + "_"+ str(reseau) + "+"  + str(leadtime) + "h.npy", allow_pickle=True)
            print("Computing quantiles and stdev of large real ensemble")

            Qrefs = np.percentile(tabref, quant, interpolation='nearest', axis=0)
            data_ref_list = [Qrefs[i] for i in range(13)]
            sdev_ref = np.std(tabref,axis=0,ddof=1)
            data_ref_list.append(sdev_ref)

            large_AROME_dir = args.output_dir+'/large_AROME'
            os.makedirs(large_AROME_dir, exist_ok=True)
            np.save(f"{large_AROME_dir}/large_AROME_{leadtime}_{param}.npy", np.concatenate([Qrefs, sdev_ref[np.newaxis,:]]))

            for quantile in Qrefs:
                clim.append((quantile.min(), quantile.max()))

            print("Plotting AROME Quantile of large real ensemble")
            plot_quantile(
                data_list=Qrefs,
                output_dir=args.output_dir,
                name_quantile='large_AROME',
                param=param,
                leadtime=leadtime,
                clim=clim,
                id_quantile_to_plot = [0,1,6,9,12],
                denom = ["Q0","Q05", "Q1", "Q5", "Q10","Q25","Q50","Q75","Q90","Q95", "Q99", "Q995", "Q100", "Sdev"]
            )

            # Keeping track of quantiles spatial means
            qref_avg = np.zeros((np.size(quant)))
            for q in range(np.size(quant)):
                qref_avg[q] = Qrefs[q].mean()

            ################################################
            ############# Small AROME Ensemble #############
            ################################################

            print("Loading small real ensemble")
            q0small = []
            q05small = []
            q1small = []
            q5small = []
            q10small = []
            q25small = []
            q50small = []
            q75small = []
            q90small = []
            q95small = []
            q99small = []
            q995small = []
            q100small = []
            sdev_small = []
            qavg_small=pd.DataFrame(columns=['leadtime','Quantiles','Init','DiffSmall', 'DiffRelSmall'])
            print("Computing quantiles and stdev of small real ensemble")
            for i in trange(nbrandinit):
                tabs = np.zeros((Nsmall,tabref.shape[1],tabref.shape[2]))
                mb = np.load(args.GAN_sample_dir + nameGAN[0] + "mb_" + str(i) + f'_{leadtime}_{args.inv_step}.npy',allow_pickle=True).astype(np.uint16)

                ### This loop should maybe be rewritten and using numpy array reindexing directly
                for k in range(Nsmall):
                    tabs[k,:,:] = tabref[int(mb[k]),:,:]

                Qsmall = np.percentile(tabs,quant,interpolation='nearest',axis=0)
                q0small.append(Qsmall[0])
                q05small.append(Qsmall[1]) 
                q1small.append(Qsmall[2])
                q5small.append(Qsmall[3])
                q10small.append(Qsmall[4])
                q25small.append(Qsmall[5])
                q50small.append(Qsmall[6])
                q75small.append(Qsmall[7])
                q90small.append(Qsmall[8])
                q95small.append(Qsmall[9])
                q99small.append(Qsmall[10])
                q995small.append(Qsmall[11])
                q100small.append(Qsmall[12])
                sdev_small.append(np.std(tabs,axis=0,ddof=1))

                for q in range(np.size(quant)):
                    qsmall_avg = np.mean(Qsmall[q])
                    newq=pd.DataFrame([[leadtime,quant[q],i,qsmall_avg-qref_avg[q],(qsmall_avg-qref_avg[q])/qref_avg[q]]],columns=['leadtime','Quantiles','Init','DiffSmall','DiffRelSmall'])
                    qavg_small=qavg_small._append(newq,ignore_index=True)
                
                Quantiles_Xtremes_avg_dir = args.output_dir+'/Quantiles_Xtremes_avg'
                os.makedirs(Quantiles_Xtremes_avg_dir, exist_ok=True)
                qavg_small.to_pickle(Quantiles_Xtremes_avg_dir + "/" + "Quantiles_Xtremes_avg_AROME_" + param + str(lag) + "_dom" + dom + "_" + "Small"  +"_"+ str(reseau) + "+" + str(leadtime) + ".pkl")
                
                small_AROME_dir = args.output_dir+'/small_AROME'
                os.makedirs(small_AROME_dir, exist_ok=True)
                np.save(f"{small_AROME_dir}/small_AROME_{leadtime}_{param}.npy",
                        np.array([np.array(q0small),np.array(q05small), np.array(q1small),
                                np.array(q5small), np.array(q10small), np.array(q25small),
                                np.array(q50small), np.array(q75small), np.array(q90small),
                                np.array(q95small), np.array(q99small), np.array(q995small),
                                np.array(q100small), np.array(sdev_small)]))
                
                
                data_list = [q0small,q05small, q1small,
                                q5small, q10small, q25small,
                                q50small, q75small, q90small,
                                q95small, q99small, q995small,
                                q100small, sdev_small]
                data_list = [np.percentile(np.array(q),50,method='nearest',axis=0) for q in data_list]

                print("Plotting AROME Quantile Small")
                
                
                plot_quantile(
                    data_list=Qsmall,
                    output_dir=args.output_dir,
                    name_quantile='small_AROME',
                    param=param,
                    leadtime=leadtime,
                    clim=clim,
                    id_quantile_to_plot = [0,1,6,9,12],
                    denom = ["Q0","Q05", "Q1", "Q5", "Q10","Q25","Q50","Q75","Q90","Q95", "Q99", "Q995", "Q100", "Sdev"]
                )

                print("Computing diff wrt to large real ensemble")

                data_diff_list = [q - qref for (q,qref) in zip(data_list, data_ref_list)]
                median_quantiles_diffsmall_dir = args.output_dir+'/median_quantiles_diffsmall'
                os.makedirs(median_quantiles_diffsmall_dir, exist_ok=True)
                np.save(f"{median_quantiles_diffsmall_dir}/median_quantiles_diffsmall_{leadtime}_{param}.npy",np.array(data_list))

            ################################################
            ############# Large GAN Ensemble #############
            ################################################

            for k in range(nbGANs):
                qavg=pd.DataFrame(columns=['leadtime','Quantiles','Init','Diff'+GANnameout[k], 'DiffRel'+GANnameout[k]])
                Qs = []
                for i in trange(nbrandinit):
                    print("Loading files containing GAN generations")
                    data = np.load(args.GAN_sample_dir + nameGAN[k] + 'samples/' + GANfilenames[k] + str(i) + '_' + str(leadtime) + f'_{args.inv_step}.npy', mmap_mode='r', allow_pickle=True)
                    tabs = np.zeros((Nsmall,tabref.shape[1],tabref.shape[2]))
                    ## this members tab is the same for all leadtimes
                    mb = np.load(args.GAN_sample_dir + nameGAN[0] + "mb_" + str(i) + f'_{leadtime}_{args.inv_step}.npy',allow_pickle=True).astype(np.uint16)

                    ### This loop should maybe be rewritten and using numpy array reindexing directly
                    for mb_idx in range(Nsmall):
                        tabs[mb_idx,:,:] = tabref[int(mb[mb_idx]),:,:]

                    print(data.shape)
                    print(tabs.shape)

                    #Extract and pre-process the data
                    if param=='t2m':
                        gan = data[:,2,:,:]
                        gan = gan - 273.15
                    elif param=='ff':
                        gan = np.sqrt(data[:,0,:,:]**2 + data[:,1,:,:]**2) * 3.6


                    if args.unbias:
                        print("unbiasing gan data wrt to conditioning AROME")

                        gan_mean = gan.mean(axis=0)
                        gan = gan - gan_mean + tabs.mean(axis=0)

                    print("Computing percentiles on GAN")
                    percentile = np.percentile(gan,quant,interpolation='nearest',axis=0)
                    Qs.append(np.concatenate([percentile,np.std(gan,axis=0,ddof=1)[np.newaxis,:]]))
                    print("Keeping track of spatial means of quantiles for GAN")
                    for q in range(np.size(quant)):
                        qm = np.mean(Qs[-1][q])
                        newq = pd.DataFrame([[leadtime,quant[q],i,qm-qref_avg[q],(qm-qref_avg[q])/qref_avg[q]]],columns=['leadtime','Quantiles','Init','Diff'+GANnameout[k],'DiffRel'+GANnameout[k]])
                        qavg = qavg._append(newq,ignore_index=True)
                        
                    if plot_id_nbrandinit == i:
                        print("Plotting GAN Quantile")
                        
                        plot_quantile(
                                data_list=percentile,
                                output_dir=args.output_dir,
                                name_quantile='GAN',
                                param=param,
                                leadtime=leadtime,
                                clim=clim,
                                id_quantile_to_plot = [0,1,6,9,12],
                                denom = ["Q0","Q05", "Q1", "Q5", "Q10","Q25","Q50","Q75","Q90","Q95", "Q99", "Q995", "Q100", "Sdev"]
                        )

                median_quantiles = np.percentile(np.array(Qs),50,interpolation='nearest',axis=0)
                median_GAN_dir = args.output_dir+'/GAN'
                os.makedirs(median_GAN_dir, exist_ok=True)
                np.save(f"{median_GAN_dir}/GAN_{GANnameout[k]}_{leadtime}_{param}.npy", np.array(Qs))
                np.save(f"{median_GAN_dir}/median_GAN_{GANnameout[k]}_{leadtime}_{param}.npy",median_quantiles)

                
                print("saving Quantiles averages")
                print(qavg.head())
                Quantiles_Xtremes_avg_dir = args.output_dir+'/Quantiles_Xtremes_avg'
                os.makedirs(Quantiles_Xtremes_avg_dir, exist_ok=True)
                qavg.to_pickle(Quantiles_Xtremes_avg_dir + "/" + "Quantiles_Xtremes_avg_GAN_" + param + str(lag) + "_dom" + dom + "_" + GANnameout[k]  +"_"+ str(reseau) + "+" + str(leadtime) + ".pkl")


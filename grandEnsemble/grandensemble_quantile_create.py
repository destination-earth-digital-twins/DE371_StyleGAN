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


if __name__=="__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument('--param', type=str, default='ff')
    parser.add_argument('--base_dir', type=str, default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/grandEnsemble/AROME/')
    parser.add_argument('--GAN_sample_dir', type=str, default='/project/home/p200177/DE_371/experiments_WP1/Grand_Ensemble/Perturbation/')
    parser.add_argument('--output_dir', type=str, default='/project/home/p200177/DE_371/experiments_WP1/Grand_Ensemble/Scores')
    parser.add_argument('--ech', type=int, default=6) # echeance de la prevision, n'importe quelle valeur entre 0 et 45h est disponible (par pas de 1h)
    parser.add_argument('--unbias', action="store_true")
    args = parser.parse_args()
    param = args.param

    ###SetUp 
    myDates = '2021100121-2021100121-PT1H' # Start-End-Step with YYYYMMDDHH format HH= reseau pour PEARO c'est 03 09 15 ou 21
    suite = "GMGI"                         # On va chercher les fichiers de l'archive oper, suite="G6CN" : on va chercher les fichiers de l'experience de recherche G6CN
    members = list(range(1,2,1))   # les 16 membres PEARO  <-- le "2" pour le 2e run audn AllMB a été chargé     
    lag = 0                      # duree cumul precip
    quant = [0,0.5,1,5,10,25,50,75,90,95,99,99.5,100]
    Nsmall = 16
    Nlbc = 25
    lGAN = True
    lvisuGAN = False # True if you want to plot individualGAN members in addition to percentiles
    lsmallens = True
    nbGANs = 1       # Nb of different GAN setups to plot
    nbrandinit = 50
    nameGAN = ["stochastic_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_False/"]
    GANfilenames = ['genFsemble_']
    GANnameout = ['cut=10_infl1.2'] 
    os.makedirs(args.output_dir, exist_ok=True)

    # Domains
    dom = 'GAN'

    # Initialisation projection
    lpercentile=True
    ldiffq=True
    lplotq=True
    lsaveq=True
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

    # Tirage nrandinit ensembles conditionneurs
    if lrandinit:
        print("tirage aléatoire des membres conditionneurs")
        mb = np.zeros((Nsmall,nbrandinit))
        for r in range(nbrandinit):
            mb[:,r] = initsmall(lstlbc,lstic,Nsmall,Nlbc)
        np.save(args.output_dir + '/' + 'nbrandinit_MBs',mb,allow_pickle=True)

    if lGAN:
        reseau = '2021-10-01T21:00:00Z'
        if ldiffq:
            #load big ensemble
            print("loading big real ensemble")
            tabref = np.load(args.base_dir + '/AllMB_domain_' + str(dom) + '_' + param + str(lag)  + "_" + suite + "_"+ str(reseau) + "+"  + str(args.ech) + "h.npy", allow_pickle=True)
            print("computing quantiles and stdev of real ensemble")

            Qrefs = np.percentile(tabref, quant, interpolation='nearest', axis=0)
            print("Plotting reference large ensemble quantiles")
            data_ref_list = [Qrefs[i] for i in range(13)]
            sdev_ref = np.std(tabref,axis=0,ddof=1)
            data_ref_list.append(sdev_ref)

            quantiles_large_dir = args.output_dir+'/quantiles_large'
            os.makedirs(quantiles_large_dir, exist_ok=True)
            np.save(f"{quantiles_large_dir}/quantiles_large_{args.ech}_{param}.npy", np.concatenate([Qrefs, sdev_ref[np.newaxis,:]]))

            prefix = args.output_dir + "/" + "MapOf_RefQuantile" + param + str(lag) + "_dom" + dom + "_" + suite
            suffix =  "_"+ str(reseau) + "+" + str(args.ech) + "h.png"
            denom = ["_Q0","_Q05", "_Q1", "_Q5", "_Q10","_Q25","_Q50","_Q75","_Q90","_Q95", "_Q99", "_Q995", "_Q100", "_Sdev"]

            print("Keeping track of quantiles spatial means")
            qref_avg = np.zeros((np.size(quant)))
            for q in range(np.size(quant)):
                qref_avg[q] = Qrefs[q].mean()

            if lsmallens:
                print("Computing stats of small ensemble")
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
                qavg_small=pd.DataFrame(columns=['args.ech','Quantiles','Init','DiffSmall', 'DiffRelSmall'])
                for i in range(nbrandinit):
                    tabs = np.zeros((Nsmall,tabref.shape[1],tabref.shape[2]))
                    mb = np.load(args.GAN_sample_dir + nameGAN[0] + "mb_" + str(i) + '_6_2000.npy',allow_pickle=True).astype(np.uint16)

                    ### This loop should maybe be rewritten and using numpy array reindexing directly
                    for k in range(Nsmall):
                        tabs[k,:,:] = tabref[int(mb[k]),:,:]

                    Qsmall = np.percentile(tabs,quant,interpolation='nearest',axis=0)
                    q0small.append(Qsmall[0])
                    q05small.append(Qsmall[1]) # np.percentile(tabref,25,interpolation='nearest',axis=0)
                    q1small.append(Qsmall[2]) # np.percentile(tabref,10,interpolation='nearest',axis=0)
                    q5small.append(Qsmall[3]) #np.percentile(tabref,0,interpolation='nearest',axis=0)
                    q10small.append(Qsmall[4])
                    q25small.append(Qsmall[5])
                    q50small.append(Qsmall[6])
                    q75small.append(Qsmall[7])
                    q90small.append(Qsmall[8])
                    q95small.append(Qsmall[9]) #)np.percentile(tabref,50,interpolation='nearest',axis=0)
                    q99small.append(Qsmall[10]) #np.percentile(tabref,75,interpolation='nearest',axis=0)
                    q995small.append(Qsmall[11]) #np.percentile(tabref,75,interpolation='nearest',axis=0)
                    q100small.append(Qsmall[12])#np
                    sdev_small.append(np.std(tabs,axis=0,ddof=1))
                    # ## I don't understand this loop
                    for q in range(np.size(quant)):
                        #print(Qsmall[q].shape)
                        qsmall_avg = np.mean(Qsmall[q])
                        newq=pd.DataFrame([[args.ech,quant[q],i,qsmall_avg-qref_avg[q],(qsmall_avg-qref_avg[q])/qref_avg[q]]],columns=['ech','Quantiles','Init','DiffSmall','DiffRelSmall'])
                        qavg_small=qavg_small.append(newq,ignore_index=True)
                
                Quantiles_Xtremes_avg_dir = args.output_dir+'/Quantiles_Xtremes_avg'
                os.makedirs(Quantiles_Xtremes_avg_dir, exist_ok=True)
                qavg_small.to_pickle(Quantiles_Xtremes_avg_dir + "/" + "Quantiles_Xtremes_avg_" + param + str(lag) + "_dom" + dom + "_" + "Small"  +"_"+ str(reseau) + "+" + str(args.ech) + ".pkl")
                
                quantiles_small_dir = args.output_dir+'/quantiles_small'
                os.makedirs(quantiles_small_dir, exist_ok=True)
                np.save(f"{quantiles_small_dir}/quantiles_small_{args.ech}_{param}.npy",
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

                if lplotq:
                    #Ici on trace les median +stdev des percentiles sur les N rand init
                    print("plotting median and stddev of percentiles on the N random init Small ensemble")


                    prefix = args.output_dir + "/" + "MapOf_MedianRandInit_" + param + str(lag) + "_dom" + dom + "_" + "Small"
                    suffix =  "_"+ str(reseau) + "+" + str(args.ech) + "h.png"
                    denom = ["_Q0","_Q05", "_Q1", "_Q5", "_Q10","_Q25","_Q50","_Q75","_Q90","_Q95", "_Q99", "_Q995", "_Q100", "_Sdev"]

                    names = [prefix + d + suffix for d in denom]


                if ldiffq:

                    print("Computing diff wrt to large real ensemble")

                    data_diff_list = [q - qref for (q,qref) in zip(data_list, data_ref_list)]
                    median_quantiles_diffsmall_dir = args.output_dir+'/median_quantiles_diffsmall'
                    os.makedirs(median_quantiles_diffsmall_dir, exist_ok=True)
                    np.save(f"{median_quantiles_diffsmall_dir}/median_quantiles_diffsmall_{args.ech}_{param}.npy",np.array(data_list))

        for k in range(nbGANs):
            qavg=pd.DataFrame(columns=['ech','Quantiles','Init','Diff'+GANnameout[k], 'DiffRel'+GANnameout[k]])
            Qs = []
            for i in range(nbrandinit):
                print("Loading files containing GAN generations")
                data = np.load(args.GAN_sample_dir + nameGAN[k] + 'samples/' + GANfilenames[k] + str(i) + '_' + str(args.ech) + '_2000.npy', mmap_mode='r', allow_pickle=True)
                tabs = np.zeros((Nsmall,tabref.shape[1],tabref.shape[2]))
                ## this members tab is the same for all leadtimes
                mb = np.load(args.GAN_sample_dir + nameGAN[0] + "mb_" + str(i) + '_6_2000.npy',allow_pickle=True).astype(np.uint16)

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


                if lpercentile:
                    print("Computing percentiles on GAN")

                    Qs.append(np.concatenate([np.percentile(gan,quant,interpolation='nearest',axis=0),np.std(gan,axis=0,ddof=1)[np.newaxis,:]]))
                    print("Keeping track of spatial means of quantiles for GAN")
                    for q in range(np.size(quant)):
                        qm = np.mean(Qs[-1][q])
                        newq = pd.DataFrame([[args.ech,quant[q],i,qm-qref_avg[q],(qm-qref_avg[q])/qref_avg[q]]],columns=['args.ech','Quantiles','Init','Diff'+GANnameout[k],'DiffRel'+GANnameout[k]])
                        qavg = qavg._append(newq,ignore_index=True)

            median_quantiles = np.percentile(np.array(Qs),50,interpolation='nearest',axis=0)
            median_quantiles_gan_dir = args.output_dir+'/quantiles_gan'
            os.makedirs(median_quantiles_gan_dir, exist_ok=True)
            np.save(f"{median_quantiles_gan_dir}/quantiles_gan_{GANnameout[k]}_{args.ech}_{param}.npy", np.array(Qs))
            np.save(f"{median_quantiles_gan_dir}/median_quantiles_gan_{GANnameout[k]}_{args.ech}_{param}.npy",median_quantiles)

            

            if ldiffq:
                print("Computing diff of quantiles median estimate with large ensemble quantiles")
                # why q50 rather than mean ?

                data_diffgan_list = [q - qref for q, qref in zip(median_quantiles, Qrefs) ]

                prefix = args.output_dir + "/" + "MapOf_diff_" + param + str(lag) + "_dom" + dom + "_" + GANnameout[k]
                suffix =  "_"+ str(reseau) + "+" + str(args.ech) + "h.png"
                denom = ["_Q0","_Q05", "_Q1", "_Q5", "_Q10","_Q25","_Q50","_Q75","_Q90","_Q95", "_Q99", "_Q995", "_Q100","_Sdev"]

                names = [prefix + d + suffix for d in denom]

            if lsaveq:
                print("saving Quantiles averages")
                print(qavg.head())
                Quantiles_avg_Xtremes_dir = args.output_dir+'/Quantiles_avg_Xtremes'
                os.makedirs(Quantiles_avg_Xtremes_dir, exist_ok=True)
                qavg.to_pickle(Quantiles_avg_Xtremes_dir + "/" + "Quantiles_avg_Xtremes" + param + str(lag) + "_dom" + dom + "_" + GANnameout[k]  +"_"+ str(reseau) + "+" + str(args.ech) + ".pkl")


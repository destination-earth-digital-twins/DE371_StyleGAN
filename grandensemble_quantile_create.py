#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Simple plot with epygram of several runs

Files are retrieved with Vortex ; they are stored in a cache defined by environment variable $MTOOLDIR

Link to epygram documentation on CNRM PC : 
file:///home/common/epygram/public/EPyGrAM/1.4.7/epygram/doc_sphinx/html/index.html

Created Created sometimes in 2021
@author: raynaud
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import epygram
import usevortex as vtx
import numpy as np
import pandas as pd
import random
from bronx.stdtypes.date import daterangex as rangex
import argparse
import rapid_sequence_plot as rsp
import pickle
import os

epygram.init_env()

rsp.activate()

parser = argparse.ArgumentParser()

parser.add_argument('--param', type=str, default='t2m')
parser.add_argument('--ech', type=int, default=42)
parser.add_argument('--unbias', action="store_true")
args = parser.parse_args()

###SetUp 
myDates = '2021100121-2021100121-PT1H' # Start-End-Step with YYYYMMDDHH format HH= reseau pour PEARO c'est 03 09 15 ou 21
suite = "GMGI"                         # On va chercher les fichiers de l'archive oper, suite="G6CN" : on va chercher les fichiers de l'experience de recherche G6CN
suite_ref = "G6CN"
cutoff = "production"
vconf = "pefrance"
members = list(range(1,2,1))   # les 16 membres PEARO  <-- le "2" pour le 2e run audn AllMB a été chargé
ech = args.ech                   # echeance de la prevision, n'importe quelle valeur entre 0 et 45h est disponible (par pas de 1h)
lag = 0                      # duree cumul precip
param = args.param #"rr" "tpw850" #Must be defined in GribID dict    ==> le parametre meteo demande
#if param = "wind" you have to set lwind (below) to True
quant = [0,0.5,1,5,10,25,50,75,90,95,99,99.5,100]
Nsmall = 16
Nlbc = 25
lwind = (args.param=='ff')
lsavePE = False # if you want to save the members on first trial
lfindpdg = False
lGAN = True
lvisuGAN = False # True if you want to plot individualGAN members in addition to percentiles
lBigEns = False # true si on veut charger; false ---> va 
lsmallens = True
nbGANs = 1       # Nb of different GAN setups to plot
nbrandinit = 50
path_GAN = '/scratch/work/brochetc/Exp_StyleGAN/Perturbation_GE/'
minmax = (0,5.0) if param=='t2m' else (0,15)
#nameGAN=['SMPCA_Latent_2_3_10_12/']#'Style_Mixing_2_3_10_12_/','Style_Mixing_0_2_9_12/']
#GANfilenames=['StyleMixing_PCA_from_z_2_3_10_12_']#'StyleMixing_2_3_10_12_','StyleMixing_0_2_9_12_']
#GANnameout=['StyleMixing_PCA_2_3_10_12']#'Style_Mixing_2_3_10_12','Style_Mixing_0_2_9_12']


nameGAN = ["stochastic_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_False_10_1.0/samples/"]

GANfilenames = ['genFsemble_']#'StyleMixing_2_3_10_12_','StyleMixing_0_2_9_12_']
GANnameout = ['cut=10_infl1.0']#,'cut=8', 'cut=10','cut=12', 'cut=14']

# choose the right option depending on the GAN file names 
#ech_GAN=np.int((ech/3)-1)
ech_GAN = ech

base_dir = "/scratch/work/brochetc/grandEnsemble/Resultats/"

unbias = args.unbias


out_dir = base_dir + f"{ech}_{param}_{GANnameout[0]}" if not unbias else base_dir + f"{ech}_{param}_{GANnameout[0]}_unbias_{unbias}"
os.makedirs(out_dir, exist_ok=True)

homedir = "/home/gmap/mrmn/exploiting_ge/"

if param=='t2m':
    sup=3
elif param=='ff':
    sup=15

# Domains
lcrop = True
dom = 'GAN'
DomID = {}
DomID["HPE"] = {"lonO":1.5,"lonE":8,"latS":42,"latN":46}
DomID["SO"] = {"lonO":0,"lonE":5,"latS":42,"latN":46}
DomID["NO"] = {"lonO":-6,"lonE":2,"latS":45,"latN":50}
DomID["GAN"] = {"lonO":0.49,"lonE":6.88,"latS":41.99,"latN":48.38}
#DomID["GAN"] = {"lonO":6.059,"lonE":2.8617,"latS":42.443,"latN":45.639}
#Corse 2 10 40 45
#######
# Look up for pdg
lat = 44.2
lon = 3.75
#Nantes 47.3°N,1.35°W 
#HPE 44.2N 3.75E

# ###GRIB ID
# ###GRIB2 help : http://intra.cnrm.meteo.fr/gws/wtg/ --> Concept --> enter name and copy dict to clipboard
# GribID = {}
# GribID["rr"] = {'parameterNumber': 65}   #RAIN GRIB2
# GribID["neige"] = {'parameterNumber': 66}
# GribID["refl"] = {"discipline":0, "parameterCategory": 16, "parameterNumber": 193, "tablesVersion" : 15, "typeOfFirstFixedSurface" : 1, "scaledValueOfFirstFixedSurface" : 0}   #REFLMAX GRIB2
# GribID["tpw850"] = {"discipline":0,"parameterCategory":0,"parameterNumber":3,"tablesVersion":15,"level":850}
# GribID["t2m"] = {"discipline" : 0,"parameterCategory": 0 , "parameterNumber" : 0, "level":2,"scaledValueOfFirstFixedSurface": 2,"typeOfFirstFixedSurface": 103,"productDefinitionTemplateNumber":1}
# GribID["u"]={"discipline": 0, "parameterCategory" : 2, "parameterNumber" : 2, "scaledValueOfFirstFixedSurface" : 10, "typeOfFirstFixedSurface" :103,  "level":10, "productDefinitionTemplateNumber" : 1}
# GribID["v"]={"discipline" : 0 , "parameterCategory" : 2, "parameterNumber": 3, "scaledValueOfFirstFixedSurface" : 10, "typeOfFirstFixedSurface" : 103, "level":10, "productDefinitionTemplateNumber" : 1}
# GribID["ISP"]={"indicatorOfParameter" : 1, "indicatorOfTypeOfLevel": 100, "level":108}

# ##Colormaps

# epygram.colormapping.register_colormap_from_json('/home/gmap/mrmn/brochetc/exploiting_ge/raf2.json')
# myraf = epygram.colormapping.ColormapHelper("raf", explicit_colorbounds=[0,20,30,40,50,60,80,100,120,140,160,180,200], normalize=True)
# epygram.colormapping.register_colormap_from_json('/home/gmap/mrmn/brochetc/exploiting_ge/t2.json')
# myt2 = epygram.colormapping.ColormapHelper("t2", explicit_colorbounds=[0,4,8,10,12,14,16,18,20,22,24,26,28], normalize=True)

# Colormaps = {}
# Colormaps['rr'] = 'rr24h'
# Colormaps['neige'] = 'rr24h'
# Colormaps['refl'] = 'rr24h'
# Colormaps['tpw850'] = 'viridis'
# Colormaps['t2m'] = None
# Colormaps['ff'] = None
# Colormaps['ISP'] = 'viridis'
# diffcolormap = 'bwr'
# my_cmh = None

# if param=='ff':
#     my_cmh=myraf
# elif param=='t2m':
#     my_cmh=myt2

# Initialisation projection
crs=None
lpercentilebig=False
lpercentile=True
ldiffq=True
lplotq=True
lsaveq=True
lstamp=False
lrandinit=False
reseaux = rangex(myDates)
print(reseaux)

def initsmall(lstbc,lstic,Ns,Nlbc):
    yic = random.sample(lstic, Ns)
    ybc = random.sample(lstbc, Ns)
    mb=np.zeros((Ns))
    # Find members corresponding to yic/ybc pairs
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
    np.save(out_dir + '/' + 'nbrandinit_MBs',mb,allow_pickle=True)

reseau= reseaux[0]
if lpercentilebig:
    print("computing percentiles of the real ensemble")
    tab=np.load(base_dir+"/"+'AllMB_'+param + str(lag)  + "_" + suite + "_"+ str(reseau) + "+"  + str(ech) + "h.npy", allow_pickle=True)
    q0 = np.percentile(tab,0.0,interpolation='nearest',axis=0)
    q05 = np.percentile(tab,0.5,interpolation='nearest',axis=0)
    q1 = np.percentile(tab,1,interpolation='nearest',axis=0)
    q5 = np.percentile(tab,5,interpolation='nearest',axis=0)
    q10 = np.percentile(tab,10,interpolation='nearest',axis=0)
    q25 = np.percentile(tab,25,interpolation='nearest',axis=0)
    q50 = np.percentile(tab,50,interpolation='nearest',axis=0)
    q75 = np.percentile(tab,75,interpolation='nearest',axis=0)
    q90 = np.percentile(tab,90,interpolation='nearest',axis=0)
    q95 = np.percentile(tab,95,interpolation='nearest',axis=0)
    q99 = np.percentile(tab,99,interpolation='nearest',axis=0)
    q995 = np.percentile(tab,99.5,interpolation='nearest',axis=0)
    q100 = np.percentile(tab,100,interpolation='nearest',axis=0)


if lGAN:
    reseau = reseaux[0]
    if ldiffq:
        #load bigens.
        print("loading big real ensemble")
        tabref = np.load(base_dir + '/AllMB_domain_' + str(dom) + '_' + param + str(lag)  + "_" + suite + "_"+ str(reseau) + "+"  + str(ech) + "h.npy", allow_pickle=True)
        print(tabref.shape)
        print(tabref.max(), tabref.min())
        print("computing quantiles and stdev of real ensemble")

        Qrefs = np.percentile(tabref, quant,interpolation='nearest',axis=0)
        q0ref = Qrefs[0]
        q05ref = Qrefs[1] # np.percentile(tabref,25,interpolation='nearest',axis=0)
        q1ref = Qrefs[2] # np.percentile(tabref,10,interpolation='nearest',axis=0)
        q5ref = Qrefs[3] #np.percentile(tabref,0,interpolation='nearest',axis=0)
        q10ref = Qrefs[4]
        q25ref = Qrefs[5]
        q50ref = Qrefs[6]
        q75ref = Qrefs[7]
        q90ref = Qrefs[8]
        q95ref = Qrefs[9] #np.percentile(tabref,50,interpolation='nearest',axis=0)
        q99ref = Qrefs[10] #np.percentile(tabref,75,interpolation='nearest',axis=0)
        q995ref = Qrefs[11] #np.percentile(tabref,75,interpolation='nearest',axis=0)
        q100ref = Qrefs[12] #np.percentile(tabref,75,interpolation='nearest',axis=0)
        sdev_ref = np.std(tabref,axis=0,ddof=1)
        print("q995 ref", q995ref.max())
        #sdev = np.std(tabref,axis=0)


        print("Plotting reference large ensemble quantiles")

        data_ref_list = [q0ref, q05ref, q1ref, q5ref, q10ref,q25ref,q50ref, q75ref,q90ref, q95ref, q99ref, q995ref,q100ref, sdev_ref]
        np.save(f"{out_dir}/quantiles_large_{ech}_{param}.npy", np.concatenate([Qrefs, sdev_ref[np.newaxis,:]]))
        #data_list = [np.percentile(q,50,interpolation='nearest',axis=0) for q in data_list]

        prefix = out_dir + "/" + "MapOf_RefQuantile" + param + str(lag) + "_dom" + dom + "_" + suite
        suffix =  "_"+ str(reseau) + "+" + str(ech) + "h.png"
        denom = ["_Q0","_Q05", "_Q1", "_Q5", "_Q10","_Q25","_Q50","_Q75","_Q90","_Q95", "_Q99", "_Q995", "_Q100", "_Sdev"]

        print("Keeping track of quantiles spatial means")
        print(np.size(quant))
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
            qavg_small=pd.DataFrame(columns=['Ech','Quantiles','Init','DiffSmall', 'DiffRelSmall'])
            for i in range(nbrandinit):
                tabs = np.zeros((Nsmall,tabref.shape[1],tabref.shape[2]))
                mb = np.load(path_GAN + nameGAN[0] + "mb_" + str(i) + '_6_1000.npy',allow_pickle=True).astype(np.uint16)

                ### This loop should maybe be rewritten and using numpy array reindexing directly
                for k in range(Nsmall):
                    tabs[k,:,:] = tabref[np.int(mb[k]),:,:]
#tabs = tabref[mb,:,:]  <<--- use

                #sdevs.append(np.std(tabs,axis=0))
                #sdevsavg[i]=np.mean(sdevs)
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
                ## I don't understand this loop
                for q in range(np.size(quant)):
                    #print(Qsmall[q].shape)
                    qsmall_avg = np.mean(Qsmall[q])
                    newq=pd.DataFrame([[ech,quant[q],i,qsmall_avg-qref_avg[q],(qsmall_avg-qref_avg[q])/qref_avg[q]]],columns=['Ech','Quantiles','Init','DiffSmall','DiffRelSmall'])
                    qavg_small=qavg_small.append(newq,ignore_index=True)
            print(qavg_small.head())
            qavg_small.to_pickle(out_dir + "/" + "Quantiles_Xtremes_avg_" + param + str(lag) + "_dom" + dom + "_" + "Small"  +"_"+ str(reseau) + "+" + str(ech) + ".pkl")
            print(len(q0small), q0small[0].shape)
            np.save(f"{out_dir}/quantiles_small_{ech}_{param}.npy",
                    np.array([np.array(q0small),np.array(q05small), np.array(q1small),
                             np.array(q5small), np.array(q10small), np.array(q25small),
                             np.array(q50small), np.array(q75small), np.array(q90small),
                             np.array(q95small), np.array(q99small), np.array(q995small),
                             np.array(q100small), np.array(sdev_small)]))
            #np.save(out_dir + "/" + "Sdev_avg_" + param + str(lag) + "_dom" + dom + "_" + "Small"  +"_"+ str(reseau) + "+" + str(ech), sdevsavg, allow_pickle=True)
            data_list = [q0small,q05small, q1small,
                             q5small, q10small, q25small,
                             q50small, q75small, q90small,
                             q95small, q99small, q995small,
                             q100small, sdev_small]
            data_list = [np.percentile(np.array(q),50,interpolation='nearest',axis=0) for q in data_list]

            if lplotq:
                #Ici on trace les median +stdev des percentiles sur les N rand init
                print("plotting median and stddev of percentiles on the N random init Small ensemble")


                prefix = out_dir + "/" + "MapOf_MedianRandInit_" + param + str(lag) + "_dom" + dom + "_" + "Small"
                suffix =  "_"+ str(reseau) + "+" + str(ech) + "h.png"
                denom = ["_Q0","_Q05", "_Q1", "_Q5", "_Q10","_Q25","_Q50","_Q75","_Q90","_Q95", "_Q99", "_Q995", "_Q100", "_Sdev"]

                names = [prefix + d + suffix for d in denom]


            if ldiffq:

                print("Computing diff wrt to large real ensemble")

                data_diff_list = [q - qref for (q,qref) in zip(data_list, data_ref_list)]
                np.save(f"{out_dir}/median_quantiles_diffsmall_{ech}_{param}.npy",np.array(data_list))

    for k in range(nbGANs):
        qavg=pd.DataFrame(columns=['Ech','Quantiles','Init','Diff'+GANnameout[k], 'DiffRel'+GANnameout[k]])
        Qs = []
        for i in range(nbrandinit):
            print("Loading files containing GAN generations")
            data = np.load(path_GAN + nameGAN[k] + GANfilenames[k] + str(i) + '_' + str(ech_GAN) + '_1000.npy', mmap_mode='r', allow_pickle=True)
            tabs = np.zeros((Nsmall,tabref.shape[1],tabref.shape[2]))
            ## this members tab is the same for all leadtimes
            mb = np.load(path_GAN + nameGAN[0] + "mb_" + str(i) + '_6_1000.npy',allow_pickle=True).astype(np.uint16)

            ### This loop should maybe be rewritten and using numpy array reindexing directly
            for mb_idx in range(Nsmall):
                tabs[mb_idx,:,:] = tabref[np.int(mb[mb_idx]),:,:]

            print(data.shape)
            print(tabs.shape)
            #Extract and pre-process the data
            if param=='t2m':
                gan = data[:,2,:,:]
                gan = gan - 273.15
            elif param=='ff':
                gan = np.sqrt(data[:,0,:,:]**2 + data[:,1,:,:]**2) * 3.6

            if lvisuGAN:
                print("plotting the 100th (random) GAN member")
                
                fig.savefig(out_dir + "/" + "MapOf_" + param  + "_dom" + dom + "_" + GANnameout[k] + "_mb100_init_"+ str(i)  +"_"+ str(reseau) + "+" + str(ech) + "h.png", dpi = 150, bbox_inches='tight')

            if unbias:
                print("unbiasing gan data wrt to conditioning AROME")

                gan_mean = gan.mean(axis=0)
                gan = gan - gan_mean + tabs.mean(axis=0)


            if lpercentile:
                print("Computing percentiles on GAN")

                Qs.append(np.concatenate([np.percentile(gan,quant,interpolation='nearest',axis=0),np.std(gan,axis=0,ddof=1)[np.newaxis,:]]))
                print("Keeping track of spatial means of quantiles for GAN")
                for q in range(np.size(quant)):
                    qm = np.mean(Qs[-1][q])
                    newq = pd.DataFrame([[ech,quant[q],i,qm-qref_avg[q],(qm-qref_avg[q])/qref_avg[q]]],columns=['Ech','Quantiles','Init','Diff'+GANnameout[k],'DiffRel'+GANnameout[k]])
                    qavg = qavg.append(newq,ignore_index=True)

        median_quantiles = np.percentile(np.array(Qs),50,interpolation='nearest',axis=0)
        np.save(f"{out_dir}/quantiles_gan_{GANnameout[k]}_{ech}_{param}.npy", np.array(Qs))
        np.save(f"{out_dir}/median_quantiles_gan_{GANnameout[k]}_{ech}_{param}.npy",median_quantiles)

        if lplotq:
            #Ici on trace les median +stdev des percentiles sur les 300 init
            print("Plotting GAN median of each quantiles")

            data_list = [q for q in median_quantiles]

            prefix = out_dir + "/" + "MapOf_MedianRandInit_" + param + str(lag) + "_dom" + dom + "_" + GANnameout[k]
            suffix =  "_"+ str(reseau) + "+" + str(ech) + "h.png"
            denom = ["_Q0","_Q05", "_Q1", "_Q5", "_Q10","_Q25","_Q50","_Q75","_Q90","_Q95", "_Q99", "_Q995", "_Q100","_Sdev"]

            names = [prefix + d + suffix for d in denom]

            ###############################################
            print("Plotting stddev of GAN quantiles")


            """data_list = [np.std(q,axis=0) for q in Qs ]

            prefix = out_dir + "/" + "MapOf_SdevRandInit_" + param + str(lag) + "_dom" + dom + "_" + GANnameout[k] 
            suffix =  "_"+ str(reseau) + "+" + str(ech) + "h.png"
            denom = ["_Q0","_Q05", "_Q1", "_Q5", "_Q10","_Q25","_Q50","_Q75","_Q90","_Q95", "_Q99", "_Q995", "_Q100"]

            names = [prefix + d + suffix for d in denom]

            field.plot_sequence(
                    data_list,names,
                    projection=crs,colormap = 'viridis',epygram_departments=True,plot_method='contourf')
                    
            plt.close()"""


        if ldiffq:
            print("Computing diff of quantiles median estimate with large ensemble quantiles")
            # why q50 rather than mean ?

            data_diffgan_list = [q - qref for q, qref in zip(median_quantiles, Qrefs) ]

            prefix = out_dir + "/" + "MapOf_diff_" + param + str(lag) + "_dom" + dom + "_" + GANnameout[k]
            suffix =  "_"+ str(reseau) + "+" + str(ech) + "h.png"
            denom = ["_Q0","_Q05", "_Q1", "_Q5", "_Q10","_Q25","_Q50","_Q75","_Q90","_Q95", "_Q99", "_Q995", "_Q100","_Sdev"]

            names = [prefix + d + suffix for d in denom]

           


        if lsaveq:
            print("saving Quantiles averages")
            print(qavg.head())
            qavg.to_pickle(out_dir + "/" + "Quantiles_avg_Xtremes" + param + str(lag) + "_dom" + dom + "_" + GANnameout[k]  +"_"+ str(reseau) + "+" + str(ech) + ".pkl")


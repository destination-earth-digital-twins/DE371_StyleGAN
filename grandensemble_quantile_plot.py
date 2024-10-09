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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--ech", type=int, default=6)
parser.add_argument("--param", type=str, choices = ['ff', 't2m'])
parser.add_argument("--output_dir", type=str, default='/project/home/p200177/DE_371/experiments_WP1/Grand_Ensemble/scores/')
args = parser.parse_args()

base_dir = args.output_dir

PATH_pkl = {'normal' : base_dir + f"{args.ech}_{args.param}_normal/",
            'normal_ub' : base_dir + f"{args.ech}_{args.param}_normal_unbias_True/",
            'random' : base_dir + f"{args.ech}_{args.param}_random/",
            'random_ub' : base_dir + f"{args.ech}_{args.param}_random_unbias_True/",
            'random_23_814' : base_dir + f"{args.ech}_{args.param}_random_23_814/",
            'random_23_814_ub' : base_dir + f"{args.ech}_{args.param}_random_23_814_unbias_True/",
            'random_ub' : base_dir + f"{args.ech}_{args.param}_random_unbias_True/",
            'normal_opt_sp' : base_dir + f"{args.ech}_{args.param}_normal_opt_sp/",
            'normal_opt_sp_ub' : base_dir + f"{args.ech}_{args.param}_normal_opt_sp_unbias_True/",
            'pca_w' : base_dir + f"{args.ech}_{args.param}_pca_w/",
            'pca_w_2.5' : base_dir + f"{args.ech}_{args.param}_pca_w_2.5/",
            'pca_w_5.0' : base_dir + f"{args.ech}_{args.param}_pca_w_5.0/",
            'mix_1.0_7_False' : base_dir + f"{args.ech}_{args.param}_mix_1.0_7_False/",
            'mix_0.5_8_False' :  base_dir + f"{args.ech}_{args.param}_mix_0.5_8_False/",
            'mix_0.5_8_True' :  base_dir + f"{args.ech}_{args.param}_mix_0.5_8_True/",
            'mix_0.5_5_True' :  base_dir + f"{args.ech}_{args.param}_mix_0.5_5_True/",
            'mix_0.5_11_True' :  base_dir + f"{args.ech}_{args.param}_mix_0.5_11_True/",
            'mix_0.5_11_False' :  base_dir + f"{args.ech}_{args.param}_mix_0.5_11_False/",
            'mix2_10' :  base_dir + f"{args.ech}_{args.param}_mix2_10/",
            'mix2_10_ub' :  base_dir + f"{args.ech}_{args.param}_mix2_10_ub/",
            'mix3_0' :  base_dir + f"{args.ech}_{args.param}_mix3_0/",
            'mix3_10' :  base_dir + f"{args.ech}_{args.param}_mix3_10/",
            'mix3_14' :  base_dir + f"{args.ech}_{args.param}_mix3_14/",
            'mix3_14_ub' :  base_dir + f"{args.ech}_{args.param}_mix3_14_unbias_True/",
            'mix3_10_ub' :  base_dir + f"{args.ech}_{args.param}_mix3_10_unbias_True/",
            'mix3_0_ub' :  base_dir + f"{args.ech}_{args.param}_mix3_0_unbias_True/",
            'cut=0' :  base_dir + f"{args.ech}_{args.param}_cut=0/",
            'cut=8' :  base_dir + f"{args.ech}_{args.param}_cut=8/",
            'pca_10_uvt' :  base_dir + f"{args.ech}_{args.param}_pca_10_uvt/",
            'cut=12' :  base_dir + f"{args.ech}_{args.param}_cut=12/",
            'cut=14' :  base_dir + f"{args.ech}_{args.param}_cut=14/",
            'cut=10' :  base_dir + f"{args.ech}_{args.param}_cut=10/",
            'cut=0_infl1.0' :  base_dir + f"{args.ech}_{args.param}_cut=0_infl1.0/",
            'cut=0_infl1.1' :  base_dir + f"{args.ech}_{args.param}_cut=0_infl1.1/",
            'cut=0_infl1.2' :  base_dir + f"{args.ech}_{args.param}_cut=0/",
            'cut=0_infl1.3' :  base_dir + f"{args.ech}_{args.param}_cut=0_infl1.3/",
            'cut=10_infl1.0' :  base_dir + f"{args.ech}_{args.param}_cut=0_infl1.0/",
            'cut=10_infl1.1' :  base_dir + f"{args.ech}_{args.param}_cut=0_infl1.1/",
            'cut=10_infl1.2' :  base_dir + f"{args.ech}_{args.param}_cut=0/",
            'cut=10_infl1.3' :  base_dir + f"{args.ech}_{args.param}_cut=0_infl1.3/",
            'cut=10_infl1.0' :  base_dir + f"{args.ech}_{args.param}_cut=0_infl1.0/",
            'cut=0_infl1.1' :  base_dir + f"{args.ech}_{args.param}_cut=0_infl1.1/",
            'cut=0_infl1.2' :  base_dir + f"{args.ech}_{args.param}_cut=0/",
            'cut=0_infl1.3' :  base_dir + f"{args.ech}_{args.param}_cut=0_infl1.3/",
            'shrink' :  base_dir + f"{args.ech}_{args.param}_shrink/",
            }


ech = args.ech
param = args.param

list_expe = ['cut=0','cut=8','cut=10','cut=12','cut=14']#, 'pca_w_5.0']
list_expe = ['cut=10','shrink']
names = {'cut=0_infl' :'cut=0' ,'cut=8' :'cut=8' ,'cut=12' :'cut=12' ,'cut=14' :'cut=14' ,'cut=10' : "cut=10", "shrink" :  "Shrink."}
list_names = [n for n in list_expe] + ['Small']

#list_data = []
list_data_xtremes = []
for exp in list_expe:
    expename = exp[:-3] if 'ub' in exp else exp
    #list_data.append(pd.read_pickle(PATH_pkl[exp] + 'Quantiles_avg_' + param + f'0_domGAN_{expename}_2021-10-01T21:00:00Z+'+ str(ech) + '.pkl'))
    list_data_xtremes.append(pd.read_pickle(PATH_pkl[exp] + 'Quantiles_avg_Xtremes' + param + f'0_domGAN_{expename}_2021-10-01T21:00:00Z+'+ str(ech) + '.pkl'))
    print(list_data_xtremes[-1].head(10))


#list_data.append(pd.read_pickle(PATH_pkl['random'] + 'Quantiles_avg_' + param + f'0_domGAN_Small_2021-10-01T21:00:00Z+'+ str(ech) + '.pkl'))
list_data_xtremes.append(pd.read_pickle(PATH_pkl['pca_10_uvt'] + 'Quantiles_Xtremes_avg_' + param + f'0_domGAN_Small_2021-10-01T21:00:00Z+'+ str(ech) + '.pkl'))

list_data_full = []
list_data_1_99 = []

merge = pd.concat(list_data_xtremes)

file_name = '_'.join(list_names)

value_vars = [f"Diff{exp}" for exp in list_names]


merge_5_95 = merge.loc[(merge['Quantiles'] >=5.0 ) & (merge['Quantiles'] <=95.0)]
dd=pd.melt(merge,
            id_vars=['Quantiles'],
            value_vars=value_vars,#['DiffSmall','Diffnormal','Diffrandom','Diffnormal_opt_sp'],
            var_name='')
sns.boxplot(x='Quantiles', y='value', data=dd, hue='')
if param=='ff':
    plt.ylim(-9,9)
elif param=='t2m':
    plt.ylim(-2,3)
plt.axhline(0.0,0,1)
plt.xlabel(f"Quantiles", fontweight='bold', fontsize=18)
plt.xticks(size=14)
plt.yticks(size=14)
plt.savefig(base_dir + 'Diffqavg_overinit_wSmall_allQ_Xtremes_' + param + '_' + str(ech) + '_' + file_name +'.pdf')
plt.clf()
plt.close()



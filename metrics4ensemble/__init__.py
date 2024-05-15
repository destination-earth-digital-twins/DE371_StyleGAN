#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:53:46 2022

@author: brochetc

metrics version 2

File include :
    
    metric2D and criterion2D APIs to be used by Trainer class from trainer_horovod
    provide a directly usable namespace from already implemented metrics

"""

import metrics4ensemble.general_metrics as GM
import metrics4ensemble.quantiles_metric as quant
import metrics4ensemble.entropy as Entropy
import metrics4ensemble.brier_score as BS
import metrics4ensemble.CRPS_calc as CRPS_calc
import metrics4ensemble.skill_spread as SP
import metrics4ensemble.rel_diagram as RD
import metrics4ensemble.rank_histogram as RH
import metrics4ensemble.bias_ensemble as BE
import metrics4ensemble.mean_bias as mb
import metrics4ensemble.spectral_variance as spvar
import metrics4ensemble.skill_spread_deviation as skspd

import numpy as np
###################### standard parameters

var_dict={'rr' : 0, 'u' : 1, 'v' : 2, 't2m' : 3 , 'orog' : 4}

var_dict_fake = {'u' : 0 , 'v' : 1, 't2m' : 2, 'orog' : 3}

vars_wo_orog = ['u', 'v', 't2m']

size = 256

######################

########################### High level APIs ##################################

class metric2D_ens():
    def __init__(self, long_name, func, variables, end_shape=(3,size,size), names = ['metric']):
        
        self.long_name = long_name
        
        self.names = names # names for each of the func's output items
        
        self.func = func #should return np.array OR tensor to benefit from parallel estimation
        
        self.end_shape = end_shape # the final shape of the tensor returned by the metric 
                                   #(except for the 'batch' or 'sample' direction)
        
        self.variables = variables # variables on which the metric is applied
        
    def selectVars(self, *args) :
        
        """
        select in the input data the variables to compute metric on
        """
        
        if len(args)==2 :
            
            real_data, fake_data = args[0]
            
            VI = [var_dict[v] for v in self.variables]
            VI_f = [var_dict_fake[v] for v in self.variables]
            
            real_data = real_data[:, VI,:,:]
            print(VI,VI_f)
            fake_data = fake_data[:, VI_f,:,:]
        
            return real_data, fake_data
        
        else :
            
            return args[0]
    

    def __call__(self, *args, **kwargs):
        
        
        ########## selecting variables check #########
        try :
            select = kwargs['select']
        except KeyError :
            
            select = True
        
        ############# selection ################
    
        if select :
        
            data = self.selectVars(args)
        
        else :
            
            data = args
            
        ########### computation ################

        reliq_kwargs ={ k :v for k,v in kwargs.items() if k!='select'}
        
        if len(data) > 2:
            
            res = np.zeros((data[0].shape[0] + 2,) + self.end_shape).astype(np.float32)
            
            for i in range(data[0].shape[0]) :
                res[i] = self.func(data[0][i], data[1][i], data[2][i],**reliq_kwargs)
            
            res[-2], res[-1] = res[:-2].mean(axis = 0), res[:-2].std(axis = 0)
            # adding the batch mean and std at the end of the vector

                        
            return res
        
        elif len(data)==2:
            
            res = np.zeros((data[0].shape[0] + 2,) + self.end_shape).astype(np.float32)
            
            for i in range(data[0].shape[0]) :
                res[i] = self.func(data[0][i], data[1][i],**reliq_kwargs)
            
            res[-2], res[-1] = res[:-2].mean(axis = 0), res[:-2].std(axis = 0)
            # adding the batch mean and std at the end of the vector

                        
            return res
        
        else :
            
            res = np.zeros((data[0].shape[0] + 2,) + self.end_shape).astype(np.float32)
            
            for i in range(data[0].shape[0]) :
                
                res[i] = self.func(data[0][i],**reliq_kwargs)
            
            res[-2], res[-1] = res[:-2].mean(axis = 0), res[:-2].std(axis = 0)
            # adding the batch mean and std at the end of the vector

            return res
   
class agg_metric2D_ens(metric2D_ens):
    def __init__(self, long_name, func, variables, end_shape=(3,size,size), names = ['metric']):
        super().__init__(long_name, func, variables, end_shape, names)
    
    def __call__(self, *args, **kwargs):

         ########## selecting variables check #########
        try :
            select = kwargs['select']
        except KeyError :
            
            select = True
        
        ############# selection ################
    
        if select :
        
            data = self.selectVars(args)
        
        else :
            
            data = args
            
        ########### computation ################

        reliq_kwargs ={ k :v for k,v in kwargs.items() if k!='select'}
        
        if len(data)==2:
            
            res = np.zeros(self.end_shape).astype(np.float32)
            
            res = self.func(data[0], data[1],**reliq_kwargs)
                        
            return res

        if len(data)==3:
            print(len(data))
            
            res = np.zeros(self.end_shape).astype(np.float32)
            
            res = self.func(data[0], data[1], data[2], **reliq_kwargs)

            return res
        
        else :
            
            res = np.zeros(self.end_shape).astype(np.float32)
                
            res = self.func(data[0],**reliq_kwargs)

            return res



##############################################################################
        ################## Metrics catalogue #####################
        
standalone_metrics = {"spectral_dev","spectral_var","quantiles", "variance"}

distance_metrics = {"quantile_score", "entropy", "bias_ensemble", "ensemble_crps",
                    "brier_score", "skill_spread", "rel_diagram", "rank_histogram",
                    "variance_diff", "mean_bias", "std_diff", "rel_std_diff"}


###################### Usable namespace #######################################


quantiles = metric2D_ens('Multiple Quantiles', 
                         quant.quantiles, vars_wo_orog, end_shape =(7,3,size,size) )

quantile_score = metric2D_ens('Quantile RMSE', 
                              quant.quantile_score, vars_wo_orog)

entropy = metric2D_ens('Added ensemble entropy', 
                       Entropy.ensemble_log_likelihood, vars_wo_orog)

ensemble_crps = metric2D_ens('Average crps',
                             CRPS_calc.ensemble_crps, vars_wo_orog, end_shape= (3,1))

brier_score = metric2D_ens('Ensemble Brier Score',
                             BS.brier_score, vars_wo_orog, end_shape = (6, 3, size, size))
skill_spread = metric2D_ens('Ensemble Brier Score',
                             SP.skill_spread, vars_wo_orog, end_shape = (2, 3, size, size))
rel_diagram = metric2D_ens('Reliability diagram',
                             RD.rel_diag, vars_wo_orog, end_shape = (6, 2, 3, size, size))
rank_histogram = metric2D_ens('Reliability diagram',
                             RH.rank_histo, vars_wo_orog, end_shape = (3,121))
bias_ensemble = metric2D_ens('Average bias', BE.bias_ens, vars_wo_orog)
variance = metric2D_ens('Variance per variable',
                        GM.simple_variance, vars_wo_orog)

variance_diff = metric2D_ens('Variance map difference', GM.variance_diff, 
                             vars_wo_orog)

std_diff = metric2D_ens('Standard variation map difference', GM.std_diff, 
                             vars_wo_orog)

mean_bias = metric2D_ens('Average bias', mb.mean_bias, vars_wo_orog)

rel_std_diff = metric2D_ens('Relative Std change', GM.relative_std_diff, vars_wo_orog)

spectral_dev = metric2D_ens('Spectral deviation', spvar.spectrum_deviation, vars_wo_orog,  end_shape = (3,45))

spectral_var = metric2D_ens('Spectral deviation', spvar.spectrum_variance, vars_wo_orog,  end_shape = (3,45))

skspdm = agg_metric2D_ens('Skill Spread', skspd.skill_spread_deviation_multi_dates,vars_wo_orog, end_shape=(3, size, size))

tsksp = agg_metric2D_ens('Skill Spread', skspd.thresholded_skill_spread_deviation_multi_dates,vars_wo_orog, end_shape=(3, size, size))

crpsdm = agg_metric2D_ens('CRPS multidates', CRPS_calc.crps_multi_dates, vars_wo_orog, end_shape=(3,))

sksp_diff = agg_metric2D_ens('CRPS ARO GAN', skspd.skill_spread_vs_aro_multi_dates, vars_wo_orog, end_shape=(3,))

crps_diff = agg_metric2D_ens('CRPS ARO GAN', CRPS_calc.crps_vs_aro_multi_dates, vars_wo_orog, end_shape=(3,))
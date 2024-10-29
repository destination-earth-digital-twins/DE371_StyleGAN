#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:43:33 2023

@author: brochetc
"""

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset



################ reference dictionary to know what variables to sample where
################ do not modify unless you know what you are doing 

var_dict={'rr' : 0, 'u' : 1, 'v' : 2, 't2m' :3 , 'orog' : 4}

################
class AromeDataset(Dataset):
    
    def __init__(self, ID_file, var_indexes, crop_indexes,
                 source_transform=None, target_transform=None, source_root=None,
                 target_root=None, config=None, mode=None):
        
        self.source_root = source_root
        self.target_root = target_root
        
        self.source_transform = source_transform
        self.target_transform = target_transform
		
        self.config = config
        
        assert mode in ['train','val','test']
        
        self.mode = mode
        
        self.labels = pd.read_csv(self.source_root+ID_file)
        
        # self.labels = self.labels[self.labels['Nature']==mode].copy()
        self.labels = self.labels.set_index(np.arange(len(self.labels)))
        
        ## portion of data to crop from (assumed fixed)
        
        self.CI = crop_indexes
        self.VI = var_indexes
        
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx, mode = 'index'):
        
        if mode == 'index' :

            source_sample_path = os.path.join(self.source_root, self.labels.iloc[idx,1])
            target_sample_path = os.path.join(self.target_root, self.labels.iloc[idx,1])
        elif mode == 'sample' :
            source_sample_path = os.path.join(self.source_root, idx)
            target_sample_path = os.path.join(self.target_root, idx)
        
        source_sample = np.float32(np.load(source_sample_path+'.npy'))\
        [self.VI, self.CI[0]:self.CI[1], self.CI[2]:self.CI[3]] 
        
        
        target_sample = np.float32(np.load(target_sample_path+'.npy'))\
        [self.VI, self.CI[0]:self.CI[1], self.CI[2]:self.CI[3]] 
        
        ## transpose to get off with transform.Normalize builtin transposition
        source_sample = source_sample.transpose((1,2,0))
        
        target_sample = target_sample.transpose((1,2,0))
        
        if self.target_transform:
            target_sample = self.target_transform(target_sample)
        
        if self.source_transform:
            source_sample = self.source_transform(source_sample)
        else :
            source_sample = target_sample
        
        
        return source_sample, target_sample

"""
class AromeDatasetRandom(Dataset):
    
    def __init__(self, data_dir, ID_file, var_indexes, full_size = (256,256), crop_size = (128,128),
                 transform = None):
        
        self.data_dir = data_dir
        self.transform = transform
        self.labels = pd.read_csv(data_dir+ID_file)
        
        ## portion of data to crop from (assumed fixed)

        self.VI = var_indexes
        self.full_size = full_size
        self.crop_size = crop_size
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        sample_path = os.path.join(self.data_dir, self.labels.iloc[idx,0])
        
        crop_X0 = np.random.randint(0, high = self.full_size[0]-self.crop_size[0])
        crop_X1 = crop_X0 + self.crop_size[0]
        crop_Y0 = np.random.randint(0, high = self.full_size[1]-self.crop_size[1])
        crop_Y1 = crop_Y0 + self.crop_size[1]
        
        sample = np.float32(np.load(sample_path+'.npy'))\
                    [self.VI, :,:]
        
        
        ## transpose to get off with transform.Normalize builtin transposition
        sample = sample.transpose((1,2,0))
        
        
        if self.transform:
            sample = self.transform(sample)
        # print(type(sample))
        ## adding coordinates as channels
        
        
        return sample"""
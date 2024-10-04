#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:44:08 2022

@authors: gandonb, rabaultj, brochetc


DataSet/DataLoader classes from Importance_Sampled images
DataSet:DataLoader classes for test samples

"""

import os
import pandas as pd

import numpy as np
import pandas as pd
import scipy.ndimage
import yaml
from filelock import FileLock
from torch import Tensor, from_numpy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor, Normalize, Compose
from gan.data.statsMasked import normalizeUnderMask
from gan.data.normalize_funcs import MultiOptionNormalize
from multiprocessing import Manager

################ reference dictionary to know what variables to sample where
################ do not modify unless you know what you are doing 

var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4, 'z500': 5, 't850': 6, 'tpw850': 7}


class DatasetCache(object):
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.manager = Manager()
        self._dict = self.manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError('Data caching is disabled and get function is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, sample, importance, pos):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (sample, importance, pos)


################
class ISDataset(Dataset):

    def __init__(self, config, dataset_handler_yaml, sample_method, variable_indices,
     transform, detransform=None, use_cache=False):
        self.config = config
        self.dataset_handler_yaml = dataset_handler_yaml
        self.sample_method = sample_method
        self.VI = variable_indices
        self.transform = transform
        self.detransform = detransform
        self.labels = pd.read_csv(f"{self.config.data_dir}{self.config.id_file}")
        # Hardcoding is generally not a good idea
        # TODO : Add these to the config instead 
        self.nb_leadtime_in_dataset=45
        self.nb_members=16
        ####################
        self.cursor_incomplete_date = 0
        # if self.config.multi_timestep_mode:
        #     if self.config.timestep_period not in [i for i in range(1,self.nb_leadtime_in_dataset+1) if 45%i==0]:
        #         raise NotImplementedError
        #     if self.config.nb_timesteps * self.config.timestep_period != self.nb_leadtime_in_dataset:
        #         print(f'Warning : {self.config.nb_timesteps} * {self.config.timestep_period} != 45')
        #         raise ValueError

        
        self.cache = DatasetCache(use_cache=use_cache)
        if use_cache:
            import resource
            resource.setrlimit(
                resource.RLIMIT_CORE,
                (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
            
        ## choosing the sampling method (either random crop or fixed coordinates)

        assert sample_method in ['random', 'coords']
        if sample_method=='coords' :
            self.CI = self.config.crop_indexes
            try:
                assert self.config.crop_indexes is not None
            except AssertionError:
                raise ValueError(f"crop_indexes are {self.CI} and sample_method is coordinates")
            try:
                assert self.CI[1] - self.CI[0] == self.config.crop_size[0]
                assert self.CI[3] - self.CI[2] == self.config.crop_size[1]
            except AssertionError :
                raise ValueError(f"Provided crop indexes ({self.CI}) should match crop size ({self.config.crop_size})")

    def __len__(self):
        if self.config.multi_timestep_mode :
            # Nb_days_in_dataset = len(self.labels)/(45*16)
            return len(self.labels) // (self.nb_leadtime_in_dataset*self.nb_members)
        else :
            return len(self.labels)

    def __getitem__(self, idx):
        if self.config.multi_timestep_mode :
            # Multi time steps :
            sample = []
            # print('################start batch################')
            for leadtime_id in np.arange(0, self.config.nb_timesteps):
                # idx is fixed for a given batch

                # The csv is organized as follow [day, leadtime, member]
                # But we want [day, member, leadtime]
                # For now a batch corresponds to a day

                # We want Multiple Leadtimes per Members : 
                #           16*self.config.timestep_period*leadtime_id
                # We need to Jump per days after iterating over all leadtimes and members of a day :
                #           ((self.nb_leadtime_in_dataset-1)*16)*((idx)//16)
                # 16 being the number of members 
                
                _idx = idx + 16*self.config.timestep_period*leadtime_id + self.cursor_incomplete_date*45*16
                if self.config.cutoff_dataset_leadtimes :
                    _idx += ((self.nb_leadtime_in_dataset-1)*16)*((idx)//16)
                # print('sample id', _idx)
                # print('Batch id: ', idx)
                # print('Leadtime h', leadtime_id*self.config.timestep_period)
                # print('Day num', _idx//((self.nb_leadtime_in_dataset)*16))
                # print('Member num', idx)
                
                if self.labels.iloc[_idx]['Date'] in ['2021-02-13T21:00:00Z', '2021-08-15T21:00:00Z', '2021-09-29T21:00:00Z', '2021-05-30T21:00:00Z']:
                    print(f"Warning : Incomplete Date : {self.labels.iloc[_idx]['Date']}, switching to next sample day")
                    self.cursor_incomplete_date += 1 # switching to next day
                    _idx = idx + 16*self.config.timestep_period*leadtime_id + ((self.nb_leadtime_in_dataset-1)*16)*((idx)//16) + self.cursor_incomplete_date*45*16
        
                # print(f"Date : {self.labels.iloc[_idx]['Date']} Member : {self.labels.iloc[_idx]['Member']} Leadtime : {self.labels.iloc[_idx]['LeadTime']}")
                
                sample_path = os.path.join(self.config.data_dir, self.labels.iloc[_idx]["Name"])
                if self.sample_method=='coords':
                    single_sample = np.float32(np.load(f"{sample_path}.npy"))[self.VI, self.CI[0]:self.CI[1], self.CI[2]:self.CI[3]] # (Nvar, H, W)
                    position = self.CI
                if self.sample_method=='random' :
                    crop_X0 = np.random.randint(0, high = self.config.full_size[0] - self.config.crop_size[0])
                    crop_X1 = crop_X0 + self.config.crop_size[0]
                    crop_Y0 = np.random.randint(0, high = self.config.full_size[1] - self.config.crop_size[1])
                    crop_Y1 = crop_Y0 + self.config.crop_size[1]
                    single_sample = np.float32(np.load(f"{sample_path}.npy"))[self.VI, crop_X0:crop_X1, crop_Y0:crop_Y1]
                    position = (crop_X0, crop_X1, crop_Y0, crop_Y1)
                if len(self.VI)>1:
                    single_sample = single_sample[np.newaxis:] # (1,Nvar,H,W) in case Nvar>1
                sample.append(single_sample)
            # print('################end batch################')
            sample = np.array(sample)
            
                
        else :
            sample_path = os.path.join(self.config.data_dir, self.labels.iloc[idx]["Name"])
            if self.sample_method=='coords':
                sample = np.float32(np.load(f"{sample_path}.npy"))[self.VI, self.CI[0]:self.CI[1], self.CI[2]:self.CI[3]]
                position = self.CI
            if self.sample_method=='random' :
                crop_X0 = np.random.randint(0, high = self.config.full_size[0] - self.config.crop_size[0])
                crop_X1 = crop_X0 + self.config.crop_size[0]
                crop_Y0 = np.random.randint(0, high = self.config.full_size[1] - self.config.crop_size[1])
                crop_Y1 = crop_Y0 + self.config.crop_size[1]
                sample = np.float32(np.load(f"{sample_path}.npy"))[self.VI, crop_X0:crop_X1, crop_Y0:crop_Y1]
                position = (crop_X0, crop_X1, crop_Y0, crop_Y1)
           
        # importance = self.labels.iloc[idx]["Importance"]
        #### IMPORTANCE_ERROR
        importance = 0

        if 'rr' in self.config.var_names: #applying transformations on rr only if selected
            for _ in range(self.dataset_handler_yaml["rr_transform"]["log_transform_iteration"]):
                if not self.config.multi_timestep_mode :
                    sample[0] = np.log(1 + sample[0])
                else :
                    sample[:,0,:,:] = np.log(1 + sample[:,0,:,:])
            if self.dataset_handler_yaml["rr_transform"]["symetrization"] and np.random.random() <= 0.5:
                if not self.config.multi_timestep_mode :
                    sample[0] = -sample[0]
                else :
                    sample[:,0,:,:] = -sample[:,0,:,:]
        ## transpose to get off with transform.Normalize builtin transposition

        if not self.config.multi_timestep_mode :

            # print(f'\n stat before normalization : \
            #       (var) (min) (mean) (max) \n \
            #       t2m{sample.min()} {sample.mean()} {sample.max()} \n\  ')
            
            sample = sample.transpose((1,2,0))  
            sample = self.transform(sample)
            
            # print(f'\n stat after normalization : \
            #       (var) (min) (mean) (max)\n \
            #       t2m{sample.min()} {sample.mean()} {sample.max()} \n\  ')
        else :
            # print(f'\n stat before normalization (shape : {np.shape(sample)}): \n \
            #       (var) (min) (mean) (max) \n \
            #        u {sample[0].min()} {sample[0].mean()} {sample[0].max()} \n \
            #        v {sample[1].min()} {sample[1].mean()} {sample[1].max()} \n \
            #        t2m {sample[2].min()} {sample[2].mean()} {sample[2].max()} \n')
            sample = sample.transpose(2,3,1,0)
            # print('after T', np.shape(sample))
            sample = np.array([self.transform(sample[:,:,:,t]) for t in range(self.config.nb_timesteps)])
            # print('after', np.shape(sample))
            
            # print(f'\n stat after normalization (shape : {np.shape(sample)}): \n \
            #       (var) (min) (mean) (max)\n \
            #       u{sample[:,:,:,0].min()} {sample[:,:,:,0].mean()} {sample[:,:,:,0].max()} \n \
            #       v{sample[:,:,:,1].min()} {sample[:,:,:,1].mean()} {sample[:,:,:,1].max()} \n \
            #       t2m{sample[:,:,:,2].min()} {sample[:,:,:,2].mean()} {sample[:,:,:,2].max()} \n \
            #             ')
            if self.config.stack_sample_along_time_and_variable :
                # [[U0, V0, T0], [U1, V1, T1], ... ]
                sample = sample.reshape((self.config.nb_timesteps*len(self.VI), single_sample.shape[-2], single_sample.shape[-1]))
                
                # [[U0,U1,U2,...], [V0,V1,V2,...], [T0,T1,T2,...]]
                # sample = np.array([sample[:,i,:,:] for i in range(len(self.VI))])

                # sample = np.vstack(sample)
                # sample should now be : (Nb_leatime*N_var, H, W)

             
        

        self.cache.cache(idx, sample, importance, position)
        return sample, importance, position


class ISData_Loader():

    def __init__(self, dataset_type, config, shuf=False):
        print(f"{dataset_type} files data loader...")
        self.config = config
        if dataset_type == "Train":
            self.batch_size = self.config.batch_size
        else:
            self.batch_size = self.config.test_samples
        self.VI = [var_dict[var] for var in self.config.var_names]
        self.sampled_indices = {var: i for i, var in enumerate(self.config.var_names)} # corresponding indices in the prepared data (after sampling)

        self.shuf = shuf #shuffle performed once per epoch

        self.dataset_handler_yaml = self.read_dataset_handler_config_file()
        self.maxs, self.mins, self.means, self.stds = self.init_normalization()

        if self.stds is not None:
            self.stds *= 1.0 / 0.95

    def read_dataset_handler_config_file(self):
        print(f"{self.config.config_dir}{self.config.dataset_handler_config}")
        with open(f"{self.config.config_dir}{self.config.dataset_handler_config}", "r") as dataset_handler_config_file:
            print(f"{self.config.config_dir}{self.config.dataset_handler_config} opened...")
            return yaml.safe_load(dataset_handler_config_file)

    def init_normalization(self):
        normalization_type = self.dataset_handler_yaml["normalization"]["type"]
        if normalization_type == "mean":
            means, stds = self.load_stat_files(normalization_type, "mean", "std")
            return None, None, means[self.VI], stds[self.VI]
        elif normalization_type == "minmax":
            maxs, mins = self.load_stat_files(normalization_type, "max", "min")
            return maxs[self.VI], mins[self.VI], None, None
        elif normalization_type == "quant":
            maxs, mins = self.load_stat_files(normalization_type, "Q99", "Q01")
            return maxs[self.VI], mins[self.VI], None, None
        print("No normalization set")
        return None, None, None, None

    def load_stat_files(self, normalization_type, str1, str2):
        mean_or_max_filename = f"{str1}_{self.dataset_handler_yaml['stat_version']}"
        mean_or_max_filename += "_log" * self.dataset_handler_yaml["rr_transform"]["log_transform_iteration"]
        std_or_min_filename = f"{str2}_{self.dataset_handler_yaml['stat_version']}"
        std_or_min_filename += "_log" * self.dataset_handler_yaml["rr_transform"]["log_transform_iteration"]
        if self.dataset_handler_yaml["normalization"]["per_pixel"]:
            mean_or_max_filename += "_ppx"
            std_or_min_filename += "_ppx"
        mean_or_max_filename += ".npy"
        std_or_min_filename += ".npy"
        print(f"Normalization set to {normalization_type}")
        means_or_maxs = np.load(f"{self.config.data_dir}{self.dataset_handler_yaml['stat_folder']}{mean_or_max_filename}").astype('float32')
        print(f"{str1} file found")
        stds_or_mins = np.load(f"{self.config.data_dir}{self.dataset_handler_yaml['stat_folder']}{std_or_min_filename}").astype('float32')
        print(f"{str2} file found")
        return means_or_maxs, stds_or_mins

    def transform(self):
        options = [ToTensor()]
        normalization = self.dataset_handler_yaml["normalization"]["type"]
        if normalization != "None":
            if 'rr' in self.config.var_names and self.dataset_handler_yaml["rr_transform"]["symetrization"]: #applying transformations on rr only if selected
                if normalization == "means":
                    self.means[0] = np.zeros_like(self.means[0])
                elif normalization == "minmax":
                    self.mins[0] = -self.maxs[0]
        options.append(MultiOptionNormalize(self.means, self.stds, self.maxs, self.mins, self.config, self.dataset_handler_yaml))
        transform = Compose(options)
        return transform

    def detransform(self):
        options = [ToTensor()]
        denormalization = self.dataset_handler_yaml["normalization"]["type"]
        if denormalization != "None":
            if 'rr' in self.config.var_names and self.dataset_handler_yaml["rr_transform"]["symetrization"]: #applying transformations on rr only if selected
                if denormalization == "means":
                    raise NotImplementedError
                    self.means[0] = np.zeros_like(self.means[0]) # TODO : Do the inverse of this
                elif denormalization == "minmax":
                    raise NotImplementedError
                    self.mins[0] = -self.maxs[0]
        options.append(MultiOptionNormalize(self.means, self.stds, self.maxs, self.mins, self.config, self.dataset_handler_yaml).denorm)
        transform = Compose(options)
        return transform

    def loader(self, world_size=None, local_rank=None, kwargs=None):

        if kwargs is not None:
            with FileLock(os.path.expanduser("~/.horovod_lock")):  # if absent, causes SIGSEGV error

                if self.config.crop_indexes is not None :
                    sample_method = 'coords'
                else:
                    sample_method = 'random'
                dataset = ISDataset(self.config, self.dataset_handler_yaml, sample_method, self.VI, self.transform(), self.detransform()) # coordinates system

        self.sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
        if kwargs is not None:
            loader = DataLoader(dataset = dataset,
                            batch_size = self.batch_size,
                            shuffle = self.shuf,
                            sampler = self.sampler,
                            drop_last = True,
                            num_workers=1,
                            **kwargs)
        else:
            loader = DataLoader(dataset = dataset,
                            batch_size = self.batch_size,
                            shuffle = self.shuf,
                            sampler = self.sampler,
                            drop_last = True,
                            num_workers=1)
        return loader

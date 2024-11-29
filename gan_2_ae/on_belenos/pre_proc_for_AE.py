import argparse
import random
import subprocess
from multiprocessing import Pool
from pathlib import Path
import os
import numpy as np
import yaml
from tqdm import tqdm


class Detransform:
    def __init__(self, config):
        self.dataset_handler_yaml = config
        self.maxs, self.mins, self.means, self.stds = self.init_normalization()
        if self.stds is not None:
            self.stds *= 1.0 / 0.95

    def init_normalization(self):
        normalization_type = self.dataset_handler_yaml['normalization']['type']
        if normalization_type == 'mean':
            means, stds = self.load_stat_files(normalization_type, 'mean', 'std')
            return None, None, means, stds
        elif normalization_type == 'minmax':
            maxs, mins = self.load_stat_files(normalization_type, 'max', 'min')
            return maxs, mins, None, None
        elif normalization_type == 'quant':
            maxs, mins = self.load_stat_files(normalization_type, 'Q99', 'Q01')
            return maxs, mins, None, None
        else:
            print('No normalization set')
            return None, None, None, None

    def load_stat_files(self, normalization_type, str1, str2):
        stat_version = self.dataset_handler_yaml['stat_version']
        log_iterations = self.dataset_handler_yaml['rr_transform']['log_transform_iteration']
        per_pixel = self.dataset_handler_yaml['normalization']['per_pixel']

        mean_or_max_filename = f'{str1}_{stat_version}'
        mean_or_max_filename += '_log' * log_iterations
        std_or_min_filename = f'{str2}_{stat_version}'
        std_or_min_filename += '_log' * log_iterations

        if per_pixel:
            mean_or_max_filename += '_ppx'
            std_or_min_filename += '_ppx'
        mean_or_max_filename += '.npy'
        std_or_min_filename += '.npy'

        print(f'Normalization set to {normalization_type}')
        stat_folder = Path(self.dataset_handler_yaml['stat_folder'])

        file_path = Path(self.dataset_handler_yaml['data_dir']) / stat_folder / Path(mean_or_max_filename)
        means_or_maxs = np.load(file_path, mmap_mode='r').astype('float32')[0]
        print(f'{str1} file loaded')

        file_path = Path(self.dataset_handler_yaml['data_dir']) / stat_folder / Path(std_or_min_filename)
        stds_or_mins = np.load(file_path, mmap_mode='r').astype('float32')[0]
        print(f'{str2} file loaded')
        return means_or_maxs, stds_or_mins

    def detransform(self, rr_npy):
        print('Detransforming...')
        norm_type = self.dataset_handler_yaml['normalization']['type']
        per_pixel = self.dataset_handler_yaml['normalization']['per_pixel']
        rr_transform = self.dataset_handler_yaml['rr_transform']
        ## Not used for now
        # if rr_transform['symetrization']:
        #     self.mins = -self.maxs
        #     self.means = np.zeros_like(self.means)
        if norm_type == 'mean':
            rr_npy = rr_npy * self.stds + self.means
        elif norm_type == 'minmax' or norm_type == 'quant':
            rr_npy = ((rr_npy + 1) / 2) * (self.maxs - self.mins) + self.mins
        ## Not used for now
        # if rr_transform['symetrization']:
        #     rr_npy[:, 0] = np.abs(rr_npy[:, 0])
        for _ in range(rr_transform['log_transform_iteration']):
            try:
                rr_npy = np.exp(rr_npy) - 1
            except RuntimeWarning as err:
                print(f'{err}, in np.exp(rr_npy) - 1.')
        ## Not used for now
        # if rr_transform['gaussian_std'] > 0:
        #     mask_no_rr = rr_npy > rr_transform['gaussian_std'] * (1 + 0.25)
        #     rr_npy[:, 0] *= mask_no_rr
        print('Detransform done.')
        return rr_npy

def process_gan_output(fake_sample_dir, step, nb_batch, nb_fake_samples, detransformer: Detransform, fake_sample_detransformed_dir, nb_batch_render=1):
    print('*' * 80)
    print(f'Processing GAN output:')
    print(f'\tfake_sample_dir: {fake_sample_dir}')
    print(f'\tstep: {step}')
    print('*' * 80)
    nb_fake_per_batch = nb_fake_samples // nb_batch

    BATCH_RENDER_SIZE = 4096

    nb_batch_to_load = BATCH_RENDER_SIZE // nb_fake_per_batch

    fake_files = np.array(list(fake_sample_dir.glob(f'*{step}*.npy')))

    for batch in range(nb_batch_render):
        print(f'Loading fake samples for batch {batch + 1}...')
        rr_npy, fake_files = load_f_sample(fake_files, nb_batch_to_load)
        rr_npy = detransformer.detransform(rr_npy)
        np.save(fake_sample_detransformed_dir / f'step_{step}_batch_{batch + 1}.npy' , rr_npy)

def load_f_sample(fake_files, nb_batch_to_load):
    selected_files = fake_files[:nb_batch_to_load]
    fake_files = np.delete(fake_files, range(nb_batch_to_load), axis=0)
    with Pool() as pool:
        rr_npy = pool.map(load_f_file, selected_files)
    return np.concatenate(rr_npy), fake_files

def load_f_file(filename):
    data = np.load(filename, mmap_mode='r+')[:, 0]
    return data


def process_arome(real_sample_dir, real_sample_export_dir, nb_batch=1):
    print('*' * 80)
    print(f'Processing AROME:')
    print(f'\treal_sample_dir: {real_sample_dir}')
    print('*' * 80)
    real_files = np.array(list(real_sample_dir.glob('_sample*.npy')))
    for batch in range(nb_batch):
        print(f'Loading real samples for batch {batch + 1}...')
        rr_npy, real_files = load_r_sample(real_files)
        print('BATCH TYPE',type(batch),batch.size)
        np.save(real_sample_export_dir / f'batch_{batch + 1}' , rr_npy)

def load_r_sample(real_files):
    selected_index = random.sample(range(len(real_files)), 4096)
    selected_files = real_files[selected_index]
    real_files = np.delete(real_files, selected_index, axis=0)
    with Pool() as pool:
        rr_npy = pool.map(load_r_file, selected_files)
    return np.array(rr_npy), real_files

def load_r_file(filename):
    data = np.load(filename, mmap_mode='r+')[0]
    return data


def str2list(li):
    if type(li)==list:
        li2 = li
        return li2
    
    elif type(li)==str:
        li2=li[1:-1].split(',')
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))

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

def str2inttuple(li):
    if type(li)==list:
        li2 =[int(p) for p in li]  
        return tuple(li2)
    
    elif type(li)==str:
        li2 = li[1:-1].split(',')
        li3 =[int(p) for p in li2]

        return tuple(li3)

    else : 
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))


if __name__ == '__main__':
    print('-' * 80)
    get_hostname = "hostname"
    output = subprocess.run(get_hostname.split(), capture_output=True, text=True)
    print(f"<{__file__}> running on {output.stdout.strip()}")
    parser = argparse.ArgumentParser()

    main_features_args = parser.add_argument_group('Main features')
    main_features_args.add_argument('--set_num', type=int, help='Set number')
    main_features_args.add_argument('--step', type=int, help='Step to throw through pipeline')
    main_features_args.add_argument('--name', type=int, help='Suffix added to the folders')

    generation_characteristics_args = parser.add_argument_group('Generation characteristics for one step')
    generation_characteristics_args.add_argument('-n', '--nb_fake_samples', type=int, default=131072, help='Number of sample generated')
    generation_characteristics_args.add_argument('--nb_batch', type=int, default=1024, help='Number of batch')
    generation_characteristics_args.add_argument('--source', type=str, choices=['AROME', 'GAN'], default='AROME', help='Select the source of the samples')
    generation_characteristics_args.add_argument('--nb_batch_render', type=int, default=1, help='Number of render batch of nb_fake_samples files')

    args = parser.parse_args()

    root_set_dir = Path(f'/project/home/p200177/DE_371/datasets/dataset_Meteo_France_rr_u_v_t2m/')
    real_sample_dir = root_set_dir / 'data/IS_rr_debug_1_1.0_0_0_0_0_0_256_large_lt'
    real_sample_export_dir = root_set_dir / f'samples_AROME_for_AE_{args.name}'
    real_sample_export_dir.mkdir(parents=True, exist_ok=True)

    if args.source == 'GAN':
        config_path = Path(f'/home/gmap/mrmn/gandonb/SAVE/styleganPNRIA/gan/configs/Set_{args.set_num}')
        config_file = config_path / Path('main_256.yaml')
        with open(config_file, 'r') as main_config_yaml:
            config = yaml.safe_load(main_config_yaml)

        lat_dim = config['ensemble']['--latent_dim'][0]
        bs = config['ensemble']['--batch_size'][0]
        lr_G = config['ensemble']['--lr_G'][0]
        lr_D = config['ensemble']['--lr_D'][0]
        size = str2inttuple(config['ensemble']['--crop_indexes'][0])[1] - str2inttuple(config['ensemble']['--crop_indexes'][0])[0]
        shape = (size, size)
        use_noise = config['ensemble']['--use_noise'][0]
        var_names = str2list(config['ensemble']['--var_names'][0])
        tanh_output = config['ensemble']['--tanh_output'][0]

        data_transform_config_filename = config_path / Path(config['ensemble']['--dataset_handler_config'][0])
        
        dir_string = Path(f"stylegan2_stylegan_dom_{size}_lat-dim_{lat_dim}_bs_{bs}_{lr_D}_{lr_G}_ch-mul_2_vars_{'_'.join(str(var_name) for var_name in var_names)}_noise_{use_noise}")
    
        fake_sample_dir = root_set_dir / f'Exp_StyleGAN/Set_{args.set_num}' / dir_string / f"Instance_1/samples{f'_{args.name}' if args.name != '' else ''}"
        fake_sample_detransformed_dir = root_set_dir / f'samples_detransformed_for_AE_{args.name}'
        fake_sample_detransformed_dir.mkdir(exist_ok=True)

        with open(data_transform_config_filename, 'r') as data_transform_config_file: 
            data_transform_config = yaml.safe_load(data_transform_config_file)
        data_transform_config['data_dir'] = config['data_dir']
        detransformer = Detransform(data_transform_config)
    
    if args.source == 'AROME':
        print('DIR',len(os.listdir(real_sample_dir)))
        process_arome(real_sample_dir, real_sample_export_dir, args.nb_batch_render)
    else:
        process_gan_output(fake_sample_dir, args.step, args.nb_batch, args.nb_fake_samples, detransformer, fake_sample_detransformed_dir, args.nb_batch_render)
    
    print(f"End of <{__file__}> execution.")
    print('-' * 80)
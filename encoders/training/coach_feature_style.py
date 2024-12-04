import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from encoders.utils import common, train_utils
from encoders.criteria import w_norm, moco_loss, scattering_loss
from encoders.criteria.SWD_loss import SwdLoss
from encoders.configs import data_configs
from encoders.datasets.arome_dataset import AromeDataset
from encoders.criteria.lpips.lpips import LPIPS
from encoders.models.feature_style_encoder.feature_style_module import FeatureStyleModule
from encoders.training.ranger import Ranger
from inversion.perceptual_loss.perceptual_loss import PerceptualLoss
from inversion.focal_frequency_loss import FocalFrequencyLoss

import numpy as np

class Coach:
    def __init__(self, config):
        self.config = config
        self.global_step = self.config.resume_step + 1 if self.config.resume_step!=0 else 0

        self.device = 'cuda:0'
        self.config.device = self.device

        # Initialize network
        self.net = FeatureStyleModule(self.config).to(self.device)
        

        # Estimate latent_avg via dense sampling if latent_avg is not available
        if self.net.latent_avg is None:
            self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()

		# Initialize loss
        
        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.config.perceptual_lambda > 0 :
            self.perceptual_loss = PerceptualLoss(config=self.config, device=self.device, multi_scale=self.config.multi_scale_perceptual_loss).to(self.device).eval()
        if self.config.ffl_lambda > 0 :
            self.ffl_loss = FocalFrequencyLoss().to(self.device)
                            
		# Initialize optimizer
        self.optimizer = self.configure_optimizers()
        self.pbar = None

    	# Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets() 
        print(f'Length of train set : {len(self.train_dataset)} | Length of test set : {len(self.test_dataset)}')

        self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.config.batch_size,
										   shuffle=True,
										   num_workers=int(self.config.workers),
										   drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.config.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.config.test_workers),
										  drop_last=True)

		# Initialize logger
        self.log_dir = os.path.join(config.exp_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        #self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(config.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.config.save_interval is None:
            self.config.save_interval = self.config.max_steps

    def train(self):
        self.net.train()
        self.pbar = tqdm(range(self.config.max_steps))
        while self.global_step < self.config.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                x, y = batch
                
                x, y = x.to(self.device).float(), y.to(self.device).float()
                
                feature_scale = min(1.0, 0.0001*self.global_step)

                img, fea, fea_recon, y_hat, y_hat_hat = self.net.forward(y, feature_scale=feature_scale)
                
                loss, loss_dict = self.calc_loss(x, y, img, fea, fea_recon, y_hat, y_hat_hat)

                loss.backward()
                self.optimizer.step()

				# Logging related
                if self.global_step==0 :
                    with open(self.log_dir+'/metrics_train.csv','w') as f :
                        f.write('Step,')
                        for key in loss_dict.keys() :
                            f.write(key+',')
                        f.write('null\n')
                
                if self.global_step % self.config.image_interval == 0: # or (self.global_step < 1000 and self.global_step % 25 == 0):
                    print('plotting for train')
                    if self.config.fake_image_on_batch :
                        b = img.size(0)//2  
                        self.parse_and_log_images(x, y, y_hat_hat[b:], title='samples/train')
                    else :
                        self.parse_and_log_images(x, y, y_hat_hat, title='samples/train')
                    
                if self.global_step % self.config.board_interval == 0:
                    self.log_metrics(self.global_step, loss_dict,  prefix='train')
                
                self.print_metrics(loss_dict, prefix='train')
                
                # Validation related
                val_loss_dict = None
                if (self.global_step % self.config.val_interval == 0 or self.global_step == self.config.max_steps):
                    print('validation')
                    val_loss_dict = self.validate() ## includes image logging !
                    
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss_total'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss_total']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.config.save_interval == 0 or self.global_step == self.config.max_steps:
                    
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)
                        
                if self.global_step == self.config.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1
                self.pbar.update(1)

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            x, y = batch
            
            with torch.no_grad():
                x, y = x.to(self.device).float(), y.to(self.device).float()
                feature_scale = min(1.0, 0.0001*self.global_step)
                
                img, fea, fea_recon, y_hat, y_hat_hat = self.net.forward(y, feature_scale=feature_scale, train=False)
                # print(y.shape, img.shape, y_hat.shape, y_hat_hat.shape)
                loss, loss_dict = self.calc_loss(x, y, img, fea, fea_recon, y_hat, y_hat_hat)
            agg_loss_dict.append(loss_dict)

			# Logging related
            if batch_idx % 50 == 0:
                if self.config.fake_image_on_batch :
                    b = img.size(0)//2  
                    self.parse_and_log_images(x, y, y_hat_hat[b:], title='samples/test', subscript='{:04d}'.format(batch_idx))
                else :
                    self.parse_and_log_images(x, y, y_hat_hat, title='samples/test', subscript='{:04d}'.format(batch_idx))

			# For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        
        if self.global_step // self.config.val_interval==1 :
        
            with open(self.log_dir+'/metrics_test.csv','w') as f :
                f.write('Step,')
                for key in loss_dict.keys() :
                    f.write(key+',')
                f.write('null\n')
        
        self.log_metrics(self.global_step, loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        params = list(self.net.encoder.parameters())
        if self.config.train_decoder:
            params += list(self.net.decoder.parameters())
        if self.config.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.config.learning_rate, weight_decay = 0.0005)
        else:
            optimizer = Ranger(params, lr=self.config.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.config.dataset_type not in data_configs.DATASETS.keys():
            raise Exception('{} is not a valid dataset_type'.format(self.config.dataset_type))
        print('Loading dataset for {}'.format(self.config.dataset_type))
        
        dataset_args = data_configs.DATASETS[self.config.dataset_type]
        path = dataset_args['train_source_root']
        transforms_dict = dataset_args['transforms'](self.config, path).get_transforms()
        train_dataset = AromeDataset('Large_lt_train_labels_1.csv',[1,2,3],
                                     [0,256,0,256],
                                     source_root=dataset_args['train_source_root'],
									  target_root=dataset_args['train_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  config=self.config,
                                      mode='train')
        
        path = dataset_args['test_source_root']
        transforms_dict = dataset_args['transforms'](self.config, path).get_transforms()
        test_dataset = AromeDataset('Large_lt_test_labels.csv',[1,2,3],
                                     [0,256,0,256], #[78,206,55,183]
                                     source_root=dataset_args['test_source_root'],
									 target_root=dataset_args['test_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 config=self.config,
                                     mode='val',
                                     length_ratio=0.01)
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def calc_loss(self, x, y, img, fea, fea_recon, y_hat, y_hat_hat, option ='train'):
        loss_dict = {}
        loss = 0.0

           
        if self.config.l2_lambda_features and (fea is not None and fea_recon is not None) :
            loss_l2_features = F.mse_loss(fea, fea_recon)
            # print('loss_l2_features', loss_l2_features)
            loss_dict['loss_l2_features'] = float(loss_l2_features)
            loss += loss_l2_features * self.config.l2_lambda_features
        if self.config.l2_lambda > 0:
            if self.config.fake_image_on_batch :
                b = img.size(0)//2     
                loss_l2 =  F.mse_loss(img[:b], y_hat[:b]) # l2 loss only on synthetic data
            else :
                loss_l2 =  F.mse_loss(img, y_hat)
            # print('loss_l2', loss_l2)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.config.l2_lambda
        
        if self.config.perceptual_lambda > 0 :
            perceptual_loss = self.perceptual_loss(img, y_hat)
            # print('perceptual_loss', perceptual_loss)
            loss_dict['perceptual_loss_concat_img_y_hat'] = float(perceptual_loss) # Understand why double loss
            loss += perceptual_loss * self.config.perceptual_lambda

            perceptual_loss = self.perceptual_loss(img, y_hat_hat)
            # print('perceptual_loss', perceptual_loss)
            loss_dict['perceptual_loss_concat_img_y_hat_y_hat'] = float(perceptual_loss)
            loss += perceptual_loss * self.config.perceptual_lambda

        if self.config.ffl_lambda > 0 :
            ffl_loss = self.ffl_loss(concat_img, y_hat)
            loss_dict['ffl_loss'] = float(ffl_loss)
            loss += ffl_loss * self.config.ffl_lambda

        if option != 'train' :
            if self.config.l2_lambda==0 :
                loss_l2 = F.mse_loss(y_hat, y)
                loss_dict['loss_l2'] = float(loss_l2)
            
            loss_mae = F.l1_loss(y_hat,y)
            loss_dict['loss_mae'] = float(loss_mae)
            

        loss_dict['loss_total'] = float(loss)
        return loss, loss_dict
    
    def log_metrics(self, step, metrics_dict, prefix):
        with open(self.log_dir+'/metrics_' + prefix + '.csv', 'a') as f:
            f.write(str(step)+',')
            for key, value in metrics_dict.items():
                f.write(str(value))
                f.write(',')
            f.write('0\n')  #last null metric
           

    def print_metrics(self, loss_dict, prefix):
        display = f'{prefix} :'
        for key, loss in loss_dict.items():
            display += f"{key}: {loss:.6f} || "
        self.pbar.set_description((display))

    def parse_and_log_images(self, x, y, y_hat, title, subscript=None, display_count=1):
        sample_data = []
        for i in range(display_count):                
            cur_sample_data = {
				'input': common.numpyfy(x[i]),
				'target': common.numpyfy(y[i]),
				'output': [common.numpyfy(y_hat[i])],
			}

            sample_data.append(cur_sample_data)
            
        self.log_images(title, sample_data=sample_data, subscript=subscript)

    def log_images(self, name, sample_data, subscript=None, log_latest=False):

        fig = common.vis_samples_diff(sample_data, self.config.n_vars, single=True)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.log_dir, name, '{}_{:04d}.png'.format(subscript, step))
        else:
            path = os.path.join(self.log_dir, name, '{:04d}.png'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
			'state_dict': self.net.state_dict(),
			'config': vars(self.config),
			'latent_avg': self.net.latent_avg
		}
        return save_dict

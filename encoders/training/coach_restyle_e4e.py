import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
from encoders.utils import common, train_utils
from encoders.criteria import moco_loss
from encoders.configs import data_configs
from encoders.datasets.arome_dataset import AromeDataset
from encoders.models.e4e import e4e
from encoders.training.ranger import Ranger
from encoders.models.e4e_modules.latent_codes_pool import LatentCodesPool
from encoders.models.e4e_modules.discriminator import LatentCodesDiscriminator
from encoders.models.encoders.restyle_e4e_encoders import ProgressiveStage
from inversion.perceptual_loss.perceptual_loss import PerceptualLoss


class Coach:
	def __init__(self, config, prev_train_checkpoint=None):
		self.config = config
		self.global_step = self.config.resume_step + 1 if self.config.resume_step!=0 else 0

		self.device = 'cuda:0'
		self.config.device = self.device

		# Initialize network
		self.net = e4e(self.config).to(self.device)

		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.net.latent_avg is None:
			self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()

		# get the image corresponding to the latent average
		self.avg_image = self.net(self.net.latent_avg.unsqueeze(0),
								  input_code=True,
								  randomize_noise=False,
								  return_latents=False,
								  average_code=True)[0]
		
		self.avg_image = self.avg_image.to(self.device).float().detach()
		
		# common.tensor2im(self.avg_image).save(os.path.join(self.config.exp_dir, 'avg_image.jpg'))

		# Initialize loss
		
		self.mse_loss = nn.MSELoss().to(self.device).eval()
		if self.config.moco_lambda > 0:
			self.moco_loss = moco_loss.MocoLoss(self.device)
		if self.config.perceptual_lambda > 0 :
			self.perceptual_loss = PerceptualLoss(config=self.config, device=self.device, multi_scale=self.config.multi_scale_perceptual_loss).to(self.device).eval()
        
		# Initialize optimizer
		self.optimizer = self.configure_optimizers()
		self.pbar = None

		# Initialize discriminator
		if self.config.w_discriminator_lambda > 0:
			self.discriminator = LatentCodesDiscriminator(512, 4).to(self.device)
			self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()), lr=config.w_discriminator_lr)
			self.real_w_pool = LatentCodesPool(self.config.w_pool_size)
			self.fake_w_pool = LatentCodesPool(self.config.w_pool_size)

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


		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(config.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.config.save_interval is None:
			self.config.save_interval = self.config.max_steps

		if prev_train_checkpoint is not None:
			self.load_from_train_checkpoint(prev_train_checkpoint)
			prev_train_checkpoint = None

	def load_from_train_checkpoint(self, ckpt):
		print('Loading previous training data...')
		self.global_step = ckpt['global_step'] + 1
		self.best_val_loss = ckpt['best_val_loss']
		self.net.load_state_dict(ckpt['state_dict'])
		if self.config.w_discriminator_lambda > 0:
			self.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
			self.discriminator_optimizer.load_state_dict(ckpt['discriminator_optimizer_state_dict'])
		if self.config.progressive_steps:
			self.check_for_progressive_training_update(is_resume_from_ckpt=True)
		print(f'Resuming training from step {self.global_step}')

	def compute_discriminator_loss(self, x):
		avg_image_for_batch = self.avg_image.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
		avg_image_for_batch.clone().detach().requires_grad_(True)
		x_input = torch.cat([x, avg_image_for_batch], dim=1)
		disc_loss_dict = {}
		if self.is_training_discriminator():
			disc_loss_dict = self.train_discriminator(x_input)
		return disc_loss_dict

	def perform_train_iteration_on_batch(self, x, y):
		y_hat, latent = None, None
		loss_dict = None
		y_hats = {idx: [] for idx in range(y.shape[0])}
		for iter in range(self.config.n_iters_per_batch):
			if iter == 0:
				avg_image_for_batch = self.avg_image.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
				x_input = torch.cat([y, avg_image_for_batch], dim=1)
				y_hat, latent = self.net.forward(x_input, latent=None, return_latents=True)
			else:
				y_hat_clone = y_hat.clone().detach().requires_grad_(True)
				latent_clone = latent.clone().detach().requires_grad_(True)
				x_input = torch.cat([y, y_hat_clone], dim=1)
				y_hat, latent = self.net.forward(x_input, latent=latent_clone, return_latents=True)

			loss, loss_dict = self.calc_loss(x, y, y_hat, latent)
			loss.backward()
			# store intermediate outputs
			for idx in range(y.shape[0]):
				y_hats[idx].append([y_hat[idx]])

		return y_hats, loss_dict

	def train(self):
		self.net.train()
		self.pbar = tqdm(range(self.config.max_steps))
		if self.config.progressive_steps:
			self.check_for_progressive_training_update()

		while self.global_step < self.config.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				
				x, y = batch
				x, y = x.to(self.device).float(), y.to(self.device).float()

				disc_loss_dict = self.compute_discriminator_loss(x)

				self.optimizer.zero_grad()
				y_hats, encoder_loss_dict = self.perform_train_iteration_on_batch(x, y)
				self.optimizer.step()

				loss_dict = {**disc_loss_dict, **encoder_loss_dict}

				# Logging related
				if self.global_step==0 :
					with open(self.log_dir+'/metrics_train.csv','w') as f :
						f.write('Step,')
						for key in loss_dict.keys() :
							f.write(key+',')
						f.write('null\n')
                
				if self.global_step % self.config.image_interval == 0 : #or (self.global_step < 1000 and self.global_step % 25 == 0):
					print('plotting for train')
					self.parse_and_log_images(x, y, y_hats, title='samples/train')

				if self.global_step % self.config.board_interval == 0:
					self.log_metrics(self.global_step, loss_dict, prefix='train')
				self.print_metrics(loss_dict, prefix='train')
				# Validation related
				val_loss_dict = None
				if (self.global_step % self.config.val_interval == 0 or self.global_step == self.config.max_steps):
					print('validation')
					val_loss_dict = self.validate()

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
				if self.config.progressive_steps:
					self.check_for_progressive_training_update()

	def perform_val_iteration_on_batch(self, x, y):
		y_hat, latent = None, None
		cur_loss_dict, id_logs = None, None
		y_hats = {idx: [] for idx in range(x.shape[0])}
		for iter in range(self.config.n_iters_per_batch):
			if iter == 0:
				avg_image_for_batch = self.avg_image.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
				x_input = torch.cat([x, avg_image_for_batch], dim=1)
			else:
				x_input = torch.cat([x, y_hat], dim=1)

			y_hat, latent = self.net.forward(x_input, latent=latent, return_latents=True)

			loss, cur_loss_dict = self.calc_loss(x, y, y_hat, latent, option = 'test')
			# store intermediate outputs
			for idx in range(x.shape[0]):
				y_hats[idx].append([y_hat[idx]])

		return y_hats, cur_loss_dict, id_logs

	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			x, y = batch
			with torch.no_grad():
				x, y = x.to(self.device).float(), y.to(self.device).float()
				# validate discriminator on batch
				avg_image_for_batch = self.avg_image.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
				x_input = torch.cat([x, avg_image_for_batch], dim=1)
				cur_disc_loss_dict = {}
				if self.is_training_discriminator():
					cur_disc_loss_dict = self.validate_discriminator(x_input)

				# validate encoder on batch
				y_hats, cur_enc_loss_dict, id_logs = self.perform_val_iteration_on_batch(x, y)

				cur_loss_dict = {**cur_disc_loss_dict, **cur_enc_loss_dict}
				agg_loss_dict.append(cur_loss_dict)

			# Logging related
			if batch_idx % 50 == 0:
				self.parse_and_log_images(x, y, y_hats,
									  title='samples/test',
									  subscript='{:04d}'.format(batch_idx))

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
		else:
			self.requires_grad(self.net.decoder, False)
		if self.config.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)
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

	def calc_loss(self, x, y, y_hat, latent, option ='train'):
		loss_dict = {}
		loss = 0.0
		id_logs = None

		# Adversarial loss
		if self.is_training_discriminator():
			loss_disc = self.compute_adversarial_loss(latent, loss_dict)
			loss += self.config.w_discriminator_lambda * loss_disc

		# delta regularization loss
		if self.config.progressive_steps and self.net.encoder.progressive_stage.value != 18:
			total_delta_loss = self.compute_delta_regularization_loss(latent, loss_dict)
			loss += self.config.delta_norm_lambda * total_delta_loss

		# similarity losses
		if self.config.l2_lambda > 0:
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict['loss_l2'] = float(loss_l2)
			loss += loss_l2 * self.config.l2_lambda

		if self.config.moco_lambda > 0:
			loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
			loss_dict['loss_moco'] = float(loss_moco)
			loss_dict['id_improve'] = float(sim_improvement)
			loss += loss_moco * self.config.moco_lambda
        	
		if self.config.perceptual_lambda > 0 :
			perceptual_loss = self.perceptual_loss(y_hat, y)
			loss_dict['perceptual_loss'] = float(perceptual_loss)
			loss += perceptual_loss * self.config.perceptual_lambda

		if option != 'train' :
			if self.config.l2_lambda==0 :
				loss_l2 = F.mse_loss(y_hat, y)
				loss_dict['loss_l2'] = float(loss_l2)
            
			loss_mae = F.l1_loss(y_hat,y)
			loss_dict['loss_mae'] = float(loss_mae)

		loss_dict['loss_total'] = float(loss)
		return loss, loss_dict

	def compute_adversarial_loss(self, latent, loss_dict):
		loss_disc = 0.
		dims_to_discriminate = self.get_dims_to_discriminate() if self.is_progressive_training() else \
			list(range(self.net.decoder.n_latent))
		for i in dims_to_discriminate:
			w = latent[:, i, :]
			fake_pred = self.discriminator(w)
			loss_disc += F.softplus(-fake_pred).mean()
		loss_disc /= len(dims_to_discriminate)
		loss_dict['encoder_discriminator_loss'] = float(loss_disc)
		return loss_disc

	def compute_delta_regularization_loss(self, latent, loss_dict):
		total_delta_loss = 0
		deltas_latent_dims = self.net.encoder.get_deltas_starting_dimensions()
		first_w = latent[:, 0, :]
		for i in range(1, self.net.encoder.progressive_stage.value + 1):
			curr_dim = deltas_latent_dims[i]
			delta = latent[:, curr_dim, :] - first_w
			delta_loss = torch.norm(delta, self.config.delta_norm, dim=1).mean()
			loss_dict[f"delta{i}_loss"] = float(delta_loss)
			total_delta_loss += delta_loss
		loss_dict['total_delta_loss'] = float(total_delta_loss)
		return total_delta_loss

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
			if isinstance(y_hat, dict):
				output = [
					[common.numpyfy(y_hat[i][iter_idx][0])]
					for iter_idx in range(len(y_hat[i]))
				]
			else:
				output = [common.numpyfy(y_hat[i])]
                
			cur_sample_data = {
				'input': common.numpyfy(x[i]),
				'target': common.numpyfy(y[i]),
				'output': output, # ouput has len = the number of iterations for residual
			}

		sample_data.append(cur_sample_data)
            
		self.log_images(title, sample_data=sample_data, subscript=subscript)
	
	def log_images(self, name, sample_data, subscript=None, log_latest=False):

		fig = common.vis_samples(sample_data, self.config.n_vars)
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

		fig = common.vis_samples_diff(sample_data, self.config.n_vars)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.log_dir, name, '{}_{:04d}_diff.png'.format(subscript, step))
		else:
			path = os.path.join(self.log_dir, name, '{:04d}_diff.png'.format(step))
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

        
	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'config': vars(self.config),
			'global_step': self.global_step,
			'optimizer': self.optimizer.state_dict(),
			'best_val_loss': self.best_val_loss,
			'latent_avg': self.net.latent_avg
		}
		if self.config.w_discriminator_lambda > 0:
			save_dict['discriminator_state_dict'] = self.discriminator.state_dict()
			save_dict['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()
		return save_dict

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Util Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

	def get_dims_to_discriminate(self):
		deltas_starting_dimensions = self.net.encoder.get_deltas_starting_dimensions()
		return deltas_starting_dimensions[:self.net.encoder.progressive_stage.value + 1]

	def is_progressive_training(self):
		return self.config.progressive_steps is not None

	def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
		for i in range(len(self.config.progressive_steps)):
			if is_resume_from_ckpt and self.global_step >= self.config.progressive_steps[i]:  # Case checkpoint
				self.net.encoder.set_progressive_stage(ProgressiveStage(i))
			if self.global_step == self.config.progressive_steps[i]:  # Case training reached progressive step
				self.net.encoder.set_progressive_stage(ProgressiveStage(i))

	@staticmethod
	def requires_grad(model, flag=True):
		for p in model.parameters():
			p.requires_grad = flag
			
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Discriminator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

	def is_training_discriminator(self):
		return self.config.w_discriminator_lambda > 0

	@staticmethod
	def discriminator_loss(real_pred, fake_pred, loss_dict):
		real_loss = F.softplus(-real_pred).mean()
		fake_loss = F.softplus(fake_pred).mean()
		loss_dict['d_real_loss'] = float(real_loss)
		loss_dict['d_fake_loss'] = float(fake_loss)
		return real_loss + fake_loss

	@staticmethod
	def discriminator_r1_loss(real_pred, real_w):
		grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_w, create_graph=True)
		grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
		return grad_penalty

	def train_discriminator(self, x):
		loss_dict = {}
		self.requires_grad(self.discriminator, True)

		with torch.no_grad():
			real_w, fake_w = self.sample_real_and_fake_latents(x)
		real_pred = self.discriminator(real_w)
		fake_pred = self.discriminator(fake_w)
		loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
		loss_dict['discriminator_loss'] = float(loss)

		self.discriminator_optimizer.zero_grad()
		loss.backward()
		self.discriminator_optimizer.step()

		# r1 regularization
		d_regularize = self.global_step % self.config.d_reg_every == 0
		if d_regularize:
			real_w = real_w.detach()
			real_w.requires_grad = True
			real_pred = self.discriminator(real_w)
			r1_loss = self.discriminator_r1_loss(real_pred, real_w)

			self.discriminator.zero_grad()
			r1_final_loss = self.config.r1 / 2 * r1_loss * self.config.d_reg_every + 0 * real_pred[0]
			r1_final_loss.backward()
			self.discriminator_optimizer.step()
			loss_dict['discriminator_r1_loss'] = float(r1_final_loss)

		# Reset to previous state
		self.requires_grad(self.discriminator, False)

		return loss_dict

	def validate_discriminator(self, x):
		with torch.no_grad():
			loss_dict = {}
			real_w, fake_w = self.sample_real_and_fake_latents(x)
			real_pred = self.discriminator(real_w)
			fake_pred = self.discriminator(fake_w)
			loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
			loss_dict['discriminator_loss'] = float(loss)
			return loss_dict

	def sample_real_and_fake_latents(self, x):
		sample_z = torch.randn(self.config.batch_size, 512, device=self.device)
		real_w = self.net.decoder.get_latent(sample_z)
		fake_w = self.net.encoder(x)
		if self.is_progressive_training():  # When progressive training, feed only unique w's
			dims_to_discriminate = self.get_dims_to_discriminate()
			fake_w = fake_w[:, dims_to_discriminate, :]
		if self.config.use_w_pool:
			real_w = self.real_w_pool.query(real_w)
			fake_w = self.fake_w_pool.query(fake_w)
		if fake_w.ndim == 3:
			fake_w = fake_w[:, 0, :]
		return real_w, fake_w

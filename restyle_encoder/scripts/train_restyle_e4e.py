"""
This file runs the main training/val loop
"""
import os
import json
import math
import sys
import pprint
import torch
from argparse import Namespace

sys.path.append(".")
sys.path.append("..")

from options.e4e_train_options import e4eTrainOptions
from restyle_encoder.training.coach_restyle_e4e import Coach


def main():
	config = e4eTrainOptions().parse()
	previous_train_ckpt = None
	if config.resume_training_from_ckpt:
		config, previous_train_ckpt = load_train_checkpoint(config)
	else:
		setup_progressive_steps(config)
		create_initial_experiment_dir(config)

	coach = Coach(config, previous_train_ckpt)
	coach.train()


def load_train_checkpoint(config):
	train_ckpt_path = config.resume_training_from_ckpt
	previous_train_ckpt = torch.load(config.resume_training_from_ckpt, map_location='cpu')
	new_config_dict = vars(config)
	config = previous_train_ckpt['config']
	config['resume_training_from_ckpt'] = train_ckpt_path
	update_new_configs(config, new_config_dict)
	pprint.pprint(config)
	config = Namespace(**config)
	if config.sub_exp_dir is not None:
		sub_exp_dir = config.sub_exp_dir
		config.exp_dir = os.path.join(config.exp_dir, sub_exp_dir)
		create_initial_experiment_dir(config)
	return config, previous_train_ckpt


def setup_progressive_steps(config):
	log_size = int(math.log(config.output_size, 2))
	num_style_layers = 2 * log_size - 2
	num_deltas = num_style_layers - 1
	if config.progressive_start is not None:  # If progressive delta training
		config.progressive_steps = [0]
		next_progressive_step = config.progressive_start
		for i in range(num_deltas):
			config.progressive_steps.append(next_progressive_step)
			next_progressive_step += config.progressive_step_every

	assert config.progressive_steps is None or is_valid_progressive_steps(config, num_style_layers), \
		"Invalid progressive training input"


def is_valid_progressive_steps(config, num_style_layers):
	return len(config.progressive_steps) == num_style_layers and config.progressive_steps[0] == 0


def create_initial_experiment_dir(config):
	os.makedirs(config.exp_dir, exist_ok=True)
	config_dict = vars(config)
	pprint.pprint(config_dict)
	with open(os.path.join(config.exp_dir, 'opt.json'), 'w') as f:
		json.dump(config_dict, f, indent=4, sort_keys=True)


def update_new_configs(ckpt_config, new_config):
	for k, v in new_config.items():
		if k not in ckpt_config:
			ckpt_config[k] = v
	if new_config['update_param_list']:
		for param in new_config['update_param_list']:
			ckpt_config[param] = new_config[param]


if __name__ == '__main__':
	main()

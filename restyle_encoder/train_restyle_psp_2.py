"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint
from glob import glob
sys.path.append(".")
sys.path.append("..")

print(sys.path)

from options.train_options import TrainOptions, createNamesFromLosses
from restyle_encoder.training.coach_restyle_psp_2 import Coach


def main(namesFromLosses=False):
    
    config = TrainOptions().parse()
    
    if namesFromLosses : 
        config.exp_dir = config.exp_dir  + createNamesFromLosses(config)
    
    os.makedirs(config.exp_dir, exist_ok=True)
    
    count = len(glob(config.exp_dir+'Instance_*'))
    
    config.exp_dir = config.exp_dir + 'Instance_{}'.format(str(count + 1))
    
    os.makedirs(config.exp_dir, exist_ok=True)

    config_dict = vars(config)
    pprint.pprint(config_dict)
    with open(os.path.join(config.exp_dir, 'opt.json'), 'w') as f:
        json.dump(config_dict, f, indent=4, sort_keys=True)

    coach = Coach(config)
    coach.train()


if __name__ == '__main__':
	main(True)

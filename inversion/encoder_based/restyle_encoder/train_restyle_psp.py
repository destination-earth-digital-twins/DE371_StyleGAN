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
from training.coach_restyle_psp import Coach


def main(namesFromLosses=False):
    
    opts = TrainOptions().parse()
    
    if namesFromLosses : 
        opts.exp_dir = opts.exp_dir  + createNamesFromLosses(opts)
    
    os.makedirs(opts.exp_dir, exist_ok=True)
    
    count = len(glob(opts.exp_dir+'Instance_*'))
    
    if opts.checkpoint_path is None :
    
        opts.exp_dir = opts.exp_dir + 'Instance_{}'.format(str(count + 1))
    
    else :
        opts.exp_dir = opts.exp_dir + 'Instance_{}'.format(str(count))
    
    os.makedirs(opts.exp_dir, exist_ok=True)

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    coach = Coach(opts)
    coach.train()


if __name__ == '__main__':
	main(True)

"""
This file runs the main training/val loop
"""
import os
import yaml
import sys
import pprint
from glob import glob

sys.path.append(".")
sys.path.append("..")

print(sys.path)

from options.e4e_train_options import e4eTrainOptions, createNamesFromLosses
from restyle_encoder.training.coach_e4e import Coach


def main(namesFromLosses=False):
    
    config = e4eTrainOptions().parse()
    
    if namesFromLosses : 
        config.exp_dir = config.exp_dir + 'e4e_training/' +  createNamesFromLosses(config)
    
    os.makedirs(config.exp_dir, exist_ok=True)
    count = len(glob(config.exp_dir+'Instance_*'))
    
    if config.checkpoint_path is None :
    
        config.exp_dir = config.exp_dir + 'Instance_{}'.format(str(count + 1))
    
    else :
        config.exp_dir = config.exp_dir + 'Instance_{}'.format(str(count))

    os.makedirs(config.exp_dir, exist_ok=True)
    
    config_dict = vars(config)
    pprint.pprint(config_dict)
    
    config_file = config.exp_dir + "/training_params.yaml"
    print("writing params config file:", config_file)
    try:
        file=open(config_file,"w")
        yaml.dump(config.__dict__,file)
    except Exception as e:
         print("unable to write params config file")
         print(e)

    # with open(os.path.join(config.exp_dir, 'opt.json'), 'w') as f:
    #     json.dump(config_dict, f, indent=4, sort_keys=True)

    coach = Coach(config)
    coach.train()


if __name__ == '__main__':
	main(True)

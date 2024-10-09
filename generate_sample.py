import os
import argparse

import torch
from torchvision import utils
from gan.model.stylegan2 import Generator
from numpy import save
from collections import OrderedDict
from tqdm import trange
import numpy as np

var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4, 'z500': 5, 't850': 6, 'tpw850': 7}

def humanbytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string.
     From : https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb"""
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2) # 1,048,576
    GB = float(KB ** 3) # 1,073,741,824
    TB = float(KB ** 4) # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)
    
def str2list(li):
    if type(li)==list:
        li2 = li
        return li2
    
    elif type(li)==str:
        li2=li[1:-1].split(',')
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))

def generate(args, g_ema, mean_latent, step):
    output_dir = args.training_dir + f'final_unconditional_samples/step={step}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mem_cuda = torch.cuda.memory_allocated(device=args.device)
    print('memory_allocated {}'.format(humanbytes(mem_cuda)))
    print(f'Generating samples for step : {step}')
    with torch.no_grad():
        g_ema.eval()
        for t in trange(args.n_batches) :
            # mem_cuda = torch.cuda.memory_allocated(device='cuda:0')
            sample_z = torch.randn(args.sample, args.latent).to(args.device)
            # print(f'memory allocated for sample z : {humanbytes(mem_cuda - torch.cuda.memory_allocated(device='cuda:0'))}')
            mem_cuda = torch.cuda.memory_allocated(device=args.device)
            # print('memory_allocated {}'.format(humanbytes(mem_cuda)))
            x_sample, w_sample, _ = g_ema([sample_z], return_latents=True, truncation=args.truncation, truncation_latent=mean_latent
	                   )
            # save(output_dir+'_w_sample_'+str(step)+'_'+str(t)+'.npy', w_sample.detach().cpu().numpy())
            if args.multi_timestep_mode :
                # no matter the configuration the data are stored as (4,256,256) (to be the same as the dataset)
                x_sample = x_sample.detach().cpu().numpy()
                # print(x_sample.shape)
                b, c, h, w = x_sample.shape
                nb_timesteps_per_variable = c//len(args.var_names)
                x_sample = x_sample.reshape((b, nb_timesteps_per_variable, len(args.var_names), h,w))
                for time_step in range(nb_timesteps_per_variable):
                    x_sample_post_processed = np.empty((b,4,h,w))
                    for batch_id in range(b):
                        _sample = x_sample[batch_id][time_step]
                        for var_id, variable_name in enumerate(args.var_names):
                            # print(np.shape(_sample[var_id]))
                            x_sample_post_processed[batch_id][var_dict[variable_name]] = _sample[var_id]
                    save(output_dir+'_x_sample_'+str(step)+'_'+str(t)+'.npy', x_sample_post_processed)
            else :
                x_sample = x_sample.detach().cpu().numpy()
                b, c, h, w = x_sample.shape
                x_sample_post_processed = np.concatenate((np.zeros((b, 1, h, w)), x_sample), axis=1)
                save(output_dir+'_x_sample_'+str(step)+'_'+str(t)+'.npy', x_sample_post_processed)

def main():

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=256,
        help="number of samples to be generated per batch",
    )
    
    parser.add_argument(
        "--n_batches", type=int, default=64, help="number of batches to be generated"
    )
    
    parser.add_argument(
        "--list_steps", type=str2list, default=[206000], help="list of training steps to be used as checkpoints"
    )

    parser.add_argument(
        "--training_dir", type=str, default="/project/home/p200177/DE_371/experiments_WP1/gan_training/exp5/" # change with your path
    )

    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    parser.add_argument('--multi_timestep_mode', action='store_true')
    parser.add_argument('--nb_timesteps', type=int, default=15)
    parser.add_argument('--g_channels', type=int, default=45)
    parser.add_argument('--timestep_period', type=int, default=3)
    parser.add_argument('--var_names', type=str2list, default=['u','v','t2m'])#, 'orog'])
    parser.add_argument('--device', type=str, default='cuda:0')#, 'orog'])

    args = parser.parse_args()
    device = args.device
    args.latent = 512
    args.n_mlp = 8
    mem_cuda = torch.cuda.memory_allocated(device=device)
    print('memory_allocated {}'.format(humanbytes(mem_cuda)))

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, nb_var=args.g_channels
    ).to(device)

    mem_g = torch.cuda.memory_allocated(device=device)-mem_cuda
    print('memory_allocated for Generator {}'.format(humanbytes(mem_g)))
    for step in args.list_steps :
        checkpoint = torch.load(args.training_dir+f'models/{str(step).zfill(6)}.pt')["g_ema"]
        if 'module' in list(checkpoint.items())[0][0]: # juglling with Pytorch versioning and different module packaging
            ckpt_adapt = OrderedDict()
            for k in checkpoint.keys():
                k0 = k[7:]
                ckpt_adapt[k0] = checkpoint[k]
            g_ema.load_state_dict(ckpt_adapt)
        else:
            g_ema.load_state_dict(checkpoint)
        if args.truncation < 1:
            with torch.no_grad():
                mean_latent = g_ema.mean_latent(args.truncation_mean)
        else:
            mean_latent = None

        generate(args, g_ema, mean_latent, step)

if __name__ == "__main__":
    main()
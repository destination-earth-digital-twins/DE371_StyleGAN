import os
import argparse

import torch
from torchvision import utils
from gan.model.stylegan2 import Generator
from numpy import save
from collections import OrderedDict
from tqdm import trange

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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with torch.no_grad():
        g_ema.eval()
        for t in trange(args.n_batches) :
            mem_cuda = torch.cuda.memory_allocated(device='cuda:0')
            sample_z = torch.randn(args.sample, args.latent).to('cuda:0')
            print(f'memory allocated for sample z : {humanbytes(mem_cuda - torch.cuda.memory_allocated(device='cuda:0'))}')
            x_sample, w_sample, _ = g_ema([sample_z], return_latents=True, truncation=args.truncation, truncation_latent=mean_latent
	                   )
            save(args.output_dir+'_x_sample_'+str(step)+'_'+str(t)+'.npy', x_sample.detach().cpu().numpy())
            save(args.output_dir+'_w_sample_'+str(step)+'_'+str(t)+'.npy', w_sample.detach().cpu().numpy())

def main():

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="number of samples to be generated per batch",
    )
    
    parser.add_argument(
        "--n_batches", type=int, default=410, help="number of batches to be generated"
    )
    
    parser.add_argument(
        "--list_steps", type=str2list, default=[32000], help="list of training steps to be used as checkpoints"
    )

    parser.add_argument(
        "--output_dir", type=str, default="/project/scratch/p200177/DE_371/victorsanchez/results/gan_training/exp_train_sequential_every_1h_channel_multiplier_6/final_unconditional_samples/" # change with your path
    )

    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/project/scratch/p200177/DE_371/victorsanchez/results/gan_training/exp_train_sequential_every_1h_channel_multiplier_6/models/", # change with your path
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=6,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8
    mem_cuda = torch.cuda.memory_allocated(device='cuda:0')
    print('memory_allocated {}'.format(humanbytes(mem_cuda)))
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, nb_var=45
    ).to('cuda:0')
    mem_g = torch.cuda.memory_allocated(device='cuda:0')-mem_cuda
    print('memory_allocated for Generator {}'.format(humanbytes(mem_g)))
    for step in args.list_steps :
        checkpoint = torch.load(args.ckpt+f'{str(step).zfill(6)}.pt')["g_ema"]
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

import torch
import argparse
import numpy as np
from collections import OrderedDict

from gan.model.stylegan2 import Generator

import perturbation.utils as utils

def load_network(params):
    ################ loading network #################
    G = Generator(params.Shape[1], 512,n_mlp=8, nb_var=params.Shape[0])
    ckpt = torch.load(params.ckpt_dir, map_location='cpu')['g_ema']

    if 'module' in list(ckpt.items())[0][0]: #juglling with Pytorch versioning and different module packaging
        ckpt_adapt = OrderedDict()
        for k in ckpt.keys():
            k0 = k[7:]
            ckpt_adapt[k0] = ckpt[k]
        G.load_state_dict(ckpt_adapt)
    else:
        G.load_state_dict(ckpt)

    G.eval()
    G = G.to(params.device)

    return G

def generate_image_from_latent(latent_vector, g_ema, device, noise=None):
    """
    Generates an image from a specific latent vector using a StyleGAN generator.

    Inputs:
        latent_vector : torch.tensor, shape B x (2 log2(H) - 2) x 512
            The latent codes to be used for image generation.
        
        g_ema : stylegan Generator
            The pre-trained StyleGAN generator.
        
        device : str or torch.device
            The device to run the computation on.
        
        noise : list of torch.Tensor or None
            Optional noise maps for each layer. If None, the generator will create its own noise.

    Returns:
        img_gen : torch.tensor, shape B x C x H x W
            The generated images.
    """
    latent_vector = latent_vector.to(device)  # Move latent vector to the specified device

    # Generate the image using the latent vector
    with torch.no_grad():
        if noise is not None:
            img_gen = g_ema([latent_vector], input_is_latent=True, noise=noise)
        else:
            img_gen = g_ema([latent_vector], input_is_latent=True)

    return img_gen[0]  # Return the generated image (first element in the output list)


if __name__=="__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument("--Shape", type=tuple, default=(3,256,256), help='size of the samples')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckpt_dir', type = str, 
                            default ='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt')
    parser.add_argument('--latent_vectors_dir', type = str, 
                        default='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_autumn')
    parser.add_argument('--output_dir',type = str, 
                        default ='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/latent_space_linear_interpolation_autumn')
    parser.add_argument("--date", type=str, default = "2021-10-02")
    parser.add_argument("--input_leadtimes", type=utils.str2intlist, default=[6,12])
    parser.add_argument("--ref_leadtimes", type=utils.str2intlist, default=[6,7,8,9,10,11,12])
    parser.add_argument("--invstep", type=int, default=1000, help="optimize iterations")
    params = parser.parse_args()

    G = load_network(params)
    input_latent_vectors = []

    weights = np.linspace(0, 1, len(params.ref_leadtimes))

    for input_leadtime in params.input_leadtimes:
        latent_vector = np.load(f"{params.latent_vectors_dir}/w_{params.date}_{input_leadtime}_{params.invstep}.npy")
        latent_vector = torch.from_numpy(latent_vector)
        input_latent_vectors.append(latent_vector)
        img_generated = generate_image_from_latent(latent_vector, G, params.device)
        np.save(
            f"{params.output_dir}/inv_{params.date}_{input_leadtime}_{params.invstep}.npy", 
            img_generated.cpu().detach().numpy()
            )
    
    for ref_leadtime in params.ref_leadtimes:
        latent_vector = np.load(f"{params.latent_vectors_dir}/w_{params.date}_{ref_leadtime}_{params.invstep}.npy")
        latent_vector = torch.from_numpy(latent_vector)
        img_generated = generate_image_from_latent(latent_vector, G, params.device)
        np.save(
            f"{params.output_dir}/inv_{params.date}_{ref_leadtime}_{params.invstep}.npy", 
            img_generated.cpu().detach().numpy()
            )

    for weight, ref_leadtime in zip(weights, params.ref_leadtimes):
        interpolated_vector = input_latent_vectors[0] * (1 - weight) + input_latent_vectors[1] * weight
        img_generated = generate_image_from_latent(interpolated_vector, G, params.device)
        np.save(
            f"{params.output_dir}/interpolated_{params.date}_{ref_leadtime}_{params.invstep}.npy", 
            img_generated.cpu().detach().numpy()
            )
"""
    for i, ref_leadtime in zip(range(len(input_latent_vectors)), params.ref_leadtimes):
        if i == len(input_latent_vectors) - 1:
            break
        intepolated_vector = (input_latent_vectors[i] + input_latent_vectors[i + 1]) / 2
        img_generated = generate_image_from_latent(intepolated_vector, G, params.device)
        np.save(
            f"{params.output_dir}/interpolated_{params.date}_{ref_leadtime}_{params.invstep}.npy", 
            img_generated.cpu().detach().numpy()
            )
"""

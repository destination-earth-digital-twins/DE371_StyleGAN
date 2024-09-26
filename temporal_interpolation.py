import torch
import argparse
import numpy as np
from collections import OrderedDict

from gan.model.stylegan2 import Generator

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
    parser.add_argument('--ckpt_dir', type = str, 
                            default ='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt')
    parser.add_argument('--device', type=str, default='cuda')
    params = parser.parse_args()


    G = load_network(params)
    latent_vector_3h = np.load("/project/scratch/p200177/DE_371/temporal_downscaling_experiments/u101834/latent_vectors/w_2021-07-16_3_2000.npy")
    latent_vector_3h = torch.from_numpy(latent_vector_3h)

    latent_vector_6h = np.load("/project/scratch/p200177/DE_371/temporal_downscaling_experiments/u101834/latent_vectors/w_2021-07-16_6_2000.npy")
    latent_vector_6h = torch.from_numpy(latent_vector_6h)

    latent_vector_9h = np.load("/project/scratch/p200177/DE_371/temporal_downscaling_experiments/u101834/latent_vectors/w_2021-07-16_9_2000.npy")
    latent_vector_9h = torch.from_numpy(latent_vector_9h)

    interpolated_latent_vector = (latent_vector_3h + latent_vector_9h) / 2

    img_generated_3h = generate_image_from_latent(latent_vector_3h, G, params.device)
    img_generated_6h = generate_image_from_latent(latent_vector_6h, G, params.device)
    img_generated_9h = generate_image_from_latent(latent_vector_9h, G, params.device)
    img_generated_6h_interpolated = generate_image_from_latent(interpolated_latent_vector, G, params.device)

    np.save("/project/scratch/p200177/DE_371/temporal_downscaling_experiments/u101834/2021-07-16_3.npy", img_generated_3h.cpu().detach().numpy())
    np.save("/project/scratch/p200177/DE_371/temporal_downscaling_experiments/u101834/2021-07-16_6.npy", img_generated_6h.cpu().detach().numpy())
    np.save("/project/scratch/p200177/DE_371/temporal_downscaling_experiments/u101834/2021-07-16_9.npy", img_generated_9h.cpu().detach().numpy())
    np.save("/project/scratch/p200177/DE_371/temporal_downscaling_experiments/u101834/2021-07-16_6_interpolated.npy", img_generated_6h.cpu().detach().numpy())
    
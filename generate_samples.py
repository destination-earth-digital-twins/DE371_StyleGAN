import os
import torch
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import time
import argparse
from torchvision.utils import save_image
from gan.model.stylegan2 import Generator
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import perturbation.utils as utils

torch.manual_seed(42)  # Ensure reproducibility


# Function to load the generator
def load_generator(ckpt_path, shape, device):
    """
    Load the StyleGAN2 generator model with given weights.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        shape (tuple): Shape of the generated samples.
        device (str): Device to load the generator on.

    Returns:
        Generator: Loaded and ready-to-use generator model.
    """
    G = Generator(shape[1], 512, n_mlp=8, nb_var=shape[0])

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')['g_ema']
    if 'module' in list(ckpt.items())[0][0]:  
        ckpt_adapt = OrderedDict((k[7:], v) for k, v in ckpt.items())
        G.load_state_dict(ckpt_adapt)
    else:
        G.load_state_dict(ckpt)

    G.eval()
    G.to(device)
    return G


# Function to generate and save samples
def generate_samples(generator, output_dir, total_samples, shape, plot=False, generate=False, cmapRR=None):
    """
    Generate samples using the generator and save them to disk.

    Args:
        generator (Generator): Pretrained generator model.
        output_dir (str): Directory to save the generated samples.
        total_samples (int): Number of samples to generate.
        shape (tuple): Shape of the samples.
        plot (bool): Whether to plot and save the generated images.
        generate (bool): Whether to save the samples as .npy files.
        cmapRR: Custom colormap for visualization.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create or load latent mean
    latent_path = os.path.join(output_dir, 'latent_mean.npy')
    if not os.path.exists(latent_path):
        latent_z = torch.empty(10000, 512).normal_().to(generator.device)
        with torch.no_grad():
            latent_mean = generator.style(latent_z).mean(dim=0).cpu()
        np.save(latent_path, latent_mean.numpy())
    else:
        latent_mean = torch.tensor(np.load(latent_path).astype(np.float32), dtype=torch.float32)

    # Progress bar for sample generation
    with tqdm(total=total_samples, desc="Samples generation", unit="sample") as pbar:
        start_time = time.time()

        for sample_idx in range(total_samples):
            # Create latent vector
            latent_z = torch.empty(1, 512).normal_().to(generator.device)
            with torch.no_grad():
                style = generator.style(latent_z)
                generated_image = generator([style])

            # Save generated sample as .npy
            if generate:
                sample_path = os.path.join(output_dir, f'sample_{sample_idx}.npy')
                np.save(sample_path, generated_image[0].squeeze(0).cpu().numpy())

            # Plot and save the image
            if plot:
                image_np = generated_image[0].squeeze(0).cpu().numpy()
                fig, axs = plt.subplots(1, shape[0], figsize=(20, 5))  # 1 row, 4 columns

                for i in range(shape[0]):
                    cmap = cmapRR if i == 0 else ('coolwarm' if i == 3 else 'viridis')
                    axs[i].imshow(image_np[i], cmap=cmap, origin='lower')
                    axs[i].axis('off')
                    axs[i].set_title(f'Variable {i+1}')

                fig.savefig(os.path.join(output_dir, f'generated_image_{sample_idx}.png'), bbox_inches='tight', dpi=150)
                plt.close(fig)

            # Update progress bar
            elapsed_time = time.time() - start_time
            pbar.update(1)
            pbar.set_postfix({"Elapsed time": f"{elapsed_time:.2f} sec"})

    print(f"Complete generation of {total_samples} samples.")


# Main entry point
if __name__ == "__main__":
    # Custom colormap
    cmapRR = colors.ListedColormap(
        ["white", "mediumpurple", "blue", "dodgerblue", "darkseagreen", "seagreen",
         "greenyellow", "yellow", "navajowhite", "sandybrown", "darkorange", "red", 
         "darkred", "black"], name='from_list', N=None
    )

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str,default="", required=True, help='Path to generator checkpoint.')
    parser.add_argument('--output_dir', type=str,default="", required=True, help='Output directory for generated samples.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for computation.')
    parser.add_argument('--var_indices', type=utils.str2intlist, default=[0, 1, 2, 3])
    parser.add_argument('--Shape', type=tuple, default=(4, 256, 256), help='Size of the samples.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plot', action='store_true', default=False, help="Plots samples or not.")
    parser.add_argument('--generate', action='store_true', default=False, help="Generate .npy samples or not.")
    parser.add_argument('--total_samples', type=int, default=16384)

    params = parser.parse_args()
    device = params.device if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(params.seed)

    # Load generator
    G = load_generator(params.ckpt_dir, params.Shape, device)

    # Generate samples
    generate_samples(
        generator=G,
        output_dir=params.output_dir,
        total_samples=params.total_samples,
        shape=params.Shape,
        plot=params.plot,
        generate=params.generate,
        cmapRR=cmapRR
    )

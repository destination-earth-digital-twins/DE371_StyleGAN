import argparse

import torch
import numpy as np

from gan.model.stylegan2 import Generator

import perturbation.utils as utils

from train_interpolator import LatentInterpolatorCorrector, load_generator, generate_image_from_latent, linear_interpolation, load_samples

def save_image(output_path, img_generated):
    np.save(output_path, img_generated.cpu().detach().numpy())

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--shape", type=tuple, default=(3,256,256), help='size of the samples')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckpt_dir', type = str,
                            default ='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt')
    parser.add_argument('--inv_dir', type = str,
                        default='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_october/inversion')
    parser.add_argument('--output_dir',type = str,
        default ='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/interpolation')
    parser.add_argument("--date", type=str, default = "2021-10-31")
    parser.add_argument("--input_leadtimes", type=utils.str2intlist, default=[6,12])
    parser.add_argument("--ref_leadtimes", type=utils.str2intlist, default=[6,7,8,9,10,11,12])
    parser.add_argument("--invstep", type=int, default=1000, help="optimize iterations")
    parser.add_argument('--model_dir',type = str,
        default ='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/interpolation_models/2024-12-05/LatentInterpolatorCorrector-1024-3-pixel050-epoch-10-2024-12-05T18_01.pt')
    args = parser.parse_args()
    print(args)
    device = args.device
    output_shape = args.shape
    ckpt_dir = args.ckpt_dir
    inv_dir = args.inv_dir
    output_dir = args.output_dir
    model_dir = args.model_dir
    input_leadtimes = args.input_leadtimes
    ref_leadtimes = args.ref_leadtimes
    date = args.date
    invstep = args.invstep

    # Load the model checkpoint
    model = LatentInterpolatorCorrector(hidden_neurons=1024, num_layers=3)
    checkpoint = torch.load(model_dir, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    sample_start, sample_end = load_samples(
        basename=f"{inv_dir}/w",
        lead_times=input_leadtimes,
        start_date=date,
        end_date=date,
        invstep=invstep
    )
    print(f"Shape of the start and end samples: {sample_start.shape}, {sample_end.shape}")
    sample_start = sample_start.to(device)
    sample_end = sample_end.to(device)

    ref_samples = load_samples(
        basename=f"{inv_dir}/w",
        lead_times=ref_leadtimes,
        start_date=date,
        end_date=date,
        invstep=invstep
    )
    print(f"Shape of the reference samples: {ref_samples.shape}")

    print("Loading the generator...")
    generator = load_generator(output_shape, ckpt_dir, device)

    print("Generating reference inverted samples...")
    for ref_sample, ref_leadtime in zip(ref_samples, ref_leadtimes):
        ref_sample = ref_sample.to(device)
        img_generated = generate_image_from_latent(ref_sample, generator, device)
        save_image(
            f"{output_dir}/inv_{date}_{ref_leadtime}_{invstep}.npy",
            img_generated
        )

    print("Generating interpolated samples...")

    timesteps = torch.linspace(0, 1, len(ref_leadtimes)).to(device)
    print(f"Selected timesteps: {timesteps}")

    for t, ref_leadtime in zip(timesteps, ref_leadtimes):
        w_latent_linear_interpolation = linear_interpolation(
            sample_start, sample_end, t
        )
        img_generated_linear = generate_image_from_latent(
            w_latent_linear_interpolation, generator, device)
        save_image(
            f"{output_dir}/interpolated_linear_{date}_{ref_leadtime}_{invstep}.npy",
            img_generated_linear
        )

        t_tensor = t.expand(sample_start.shape[0], 1)
        with torch.no_grad():
            w_model = model(sample_start, sample_end, t_tensor)
        img_generated_nn = generate_image_from_latent(
            w_model, generator, device)
        save_image(
            f"{output_dir}/interpolated_NN_{date}_{ref_leadtime}_{invstep}.npy",
            img_generated_nn
        )
    print("Interpolation finished!")

if __name__=="__main__" :
    main()

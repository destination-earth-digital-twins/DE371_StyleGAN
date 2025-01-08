import os
import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np

import time_interpolation.models as models

from time_interpolation.training import load_generator, generate_image_from_latent, linear_interpolation, get_mse
from time_interpolation.dataset import InterpolatorDataset

def interpolate(dataloader, model, generator, device, args):
    mean_inverted_mse = torch.zeros(3).to(device)
    mean_phys_linear_interpolation_mse = torch.zeros(3).to(device)
    mean_latent_linear_interpolation_mse = torch.zeros(3).to(device)
    mean_latent_nn_interpolation_mse = torch.zeros(3).to(device)

    num_batches = len(dataloader)

    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            date, t_start, t_end, t_int = dataloader.dataset.indices[index]
            print(f"Date: {date}, t_start={t_start}, t_end={t_end}, t_int={t_int}")
            batch = [x.to(device).view(-1, *x.shape[2:]) for x in batch]
            w_start, w_end, t, w_t, r_start, r_end, r_t = batch

            w_interpolated = model(w_start, w_end, t).to(device)
            w_latent_linear_interpolation = linear_interpolation(w_start, w_end, t[0])

            r_inverted = generate_image_from_latent(w_t, generator).to(device)
            r_latent_nn_interpolation = generate_image_from_latent(w_interpolated, generator).to(device)
            r_latent_linear_interpolation = generate_image_from_latent(w_latent_linear_interpolation, generator)
            r_phys_interpolated = linear_interpolation(r_start, r_end, t)
            
            # Compute MSE metrics
            inverted_mse = get_mse(r_t, r_inverted, device)
            phys_linear_interpolation_mse = get_mse(r_t, r_phys_interpolated, device)
            latent_linear_interpolation_mse = get_mse(r_t, r_latent_linear_interpolation, device)
            latent_nn_interpolation_mse = get_mse(r_t, r_latent_nn_interpolation, device)

            mean_inverted_mse += inverted_mse
            mean_phys_linear_interpolation_mse += phys_linear_interpolation_mse
            mean_latent_linear_interpolation_mse += latent_linear_interpolation_mse
            mean_latent_nn_interpolation_mse += latent_nn_interpolation_mse

            print(f"Inversion MSE (1000x): {inverted_mse * 1E3}")
            print(f"Physical linear interpolation MSE (1000x): {phys_linear_interpolation_mse * 1E3}")
            print(f"Latent linear interpolation MSE (1000x): {latent_linear_interpolation_mse * 1E3}")
            print(f"Latent NN interpolation MSE (1000x): {latent_nn_interpolation_mse * 1E3}")

            relative_improvement = 100 * (phys_linear_interpolation_mse - latent_nn_interpolation_mse) / phys_linear_interpolation_mse
            print(f"Relative NN interpolation improvement (compared to physical linear, %): {relative_improvement}\n")

            output_dir = args.base_dir + args.output_dir

            save_image(f"{output_dir}/inv_{date}_{t_int}_{args.invstep}.npy", r_inverted)
            save_image(f"{output_dir}/interpolated_latent_linear_{date}_{t_int}_{args.invstep}.npy", r_latent_linear_interpolation)
            save_image(f"{output_dir}/interpolated_NN_{date}_{t_int}_{args.invstep}.npy", r_latent_nn_interpolation)
            save_image(f"{output_dir}/interpolated_phys_linear_{date}_{t_int}.npy", r_phys_interpolated)
        
    print("Overall metrics:")
    print(f"Inversion MSE (1000x): {mean_inverted_mse / num_batches * 1E3}")
    print(f"Physical linear interpolation MSE (1000x): {mean_phys_linear_interpolation_mse / num_batches * 1E3}")
    print(f"Latent linear interpolation MSE (1000x): {mean_latent_linear_interpolation_mse / num_batches * 1E3}")
    print(f"Latent NN interpolation MSE (1000x): {mean_latent_nn_interpolation_mse / num_batches * 1E3}")

    relative_improvement = 100 * (mean_phys_linear_interpolation_mse - mean_latent_nn_interpolation_mse) / mean_phys_linear_interpolation_mse
    print(f"Mean relative NN interpolation improvement (compared to physical linear, %): {relative_improvement}\n")

def save_image(output_path, img_generated):
    if not os.path.exists(output_path):
        np.save(output_path, img_generated.cpu().detach().numpy())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=tuple, default=(3,256,256), help='size of the samples')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for the dataloader.")
    parser.add_argument('--base_dir', type=str, default='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/')
    parser.add_argument('--ckpt_dir', type=str,
                            default ='/project/home/p200177/DE_371/resources/models/trained_generator/000024.pt')
    parser.add_argument('--inv_dir', type=str, default='inversion_october/inversion/')
    parser.add_argument('--pack_dir', type=str, default='inversion_october/pack/')
    parser.add_argument('--output_dir',type=str, default ='interpolation')
    parser.add_argument("--start_date", type=str, default = "2021-10-01")
    parser.add_argument("--end_date", type=str, default = "2021-10-07")
    parser.add_argument("--invstep", type=int, default=1000, help="optimize iterations")
    parser.add_argument('--model_name', type=str, default='LatentCodeInterpolator', help="Name of the model.")
    parser.add_argument('--model_path',type=str,
        default ='interpolation_models/2024-12-12/LatentInterpolatorCorrector-1024-3-pixel1000-2020-2021-period-epoch-20-2024-12-12T01_08.pt')
    parser.add_argument('--num_neurons', type=int, default=1024, help="Number of hidden neurons.")
    parser.add_argument('--num_layers', type=int, default=3, help="Number of hidden layers.")
    parser.add_argument('--normalization', type=str, default="Layer", help="Layer, Batch normalization or none.")
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout probability.")

    args = parser.parse_args()
    print(args)
    device = args.device
    num_workers = args.num_workers
    output_shape = args.shape
    ckpt_dir = args.ckpt_dir
    base_dir = args.base_dir
    inv_dir = base_dir + args.inv_dir
    pack_dir = base_dir + args.pack_dir
    model_name = args.model_name
    model_path = base_dir + args.model_path
    start_date = args.start_date
    end_date = args.end_date

    # Load the model checkpoint
    model_classes = {
        "LatentCodeInterpolator": models.LatentCodeInterpolator,
        "LatentCodeInterpolatorCorrector": models.LatentCodeInterpolatorCorrector,
        "StyleVectorInterpolator": models.StyleVectorInterpolator,
        "StyleVectorInterpolatorCorrector": models.StyleVectorInterpolatorCorrector
    }

    # Initialize model, loss function and optimizer
    if model_name in model_classes:
        model = model_classes[model_name](args).to(device)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    print(f"Model architecture: {model}\n")

    if not os.path.exists(model_path):
        print(f"Could not find {model_path}!")
        return
        
    checkpoint = torch.load(model_path, map_location=device)

    # Remove "module." prefix from keys
    # Necessary to run the model trained in DDT mode (if running on 1 GPU)
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    # 21:00, 03:00, 09:00, 15:00 as input
    dataset = InterpolatorDataset(
        start_date=start_date,
        end_date=end_date,
        latent_basepath=f"{inv_dir}w",
        real_basepath=f"{pack_dir}Rsemble",
        leadtimes=np.arange(1, 26, 6),
        dt=6, fmt='npy', include_input_leadtimes=False)
    
    intepolation_dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)

    print("Loading the generator...")
    generator = load_generator(output_shape, ckpt_dir, device)

    interpolate(intepolation_dataloader, model, generator, device, args)
    print("Interpolation finished!")

if __name__=="__main__" :
    main()

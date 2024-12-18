import os
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from inversion.perceptual_loss.perceptual_loss import PerceptualLoss
import perturbation.utils as utils
import time_interpolation.models as models
from time_interpolation.training import combined_loss, load_generator, train_loop, test_loop
from time_interpolation.dataset import InterpolatorDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device', type=str, default='cuda', help="Device to use for computation ('cpu', 'cuda', etc). Default is 'cuda'."
    )
    parser.add_argument(
        '--num_workers', type=int, default=0, help="Number of workers for the dataloader."
    )
    parser.add_argument(
        '--inv_dir', type=str, default='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_hourly/inversion/'
    )
    parser.add_argument(
        '--pack_dir', type=str, default='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_hourly/pack/'
    )
    parser.add_argument(
        '--inv_dir_val', type=str, default='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_hourly/validation/inversion/'
    )
    parser.add_argument(
        '--pack_dir_val', type=str, default='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_hourly/validation/pack/'
    )
    parser.add_argument(
        '--model_name', type=str, default='LatentInterpolator', help="Name of the model."
    )
    parser.add_argument(
        '--training_description', type=str, default='training', help="Description of the specific training."
    )
    parser.add_argument(
        '--num_neurons', type=int, default=512, help="Number of hidden neurons."
    )
    parser.add_argument(
        '--num_layers', type=int, default=3, help="Number of hidden layers."
    )
    parser.add_argument(
        '--normalization', type=str, default="Layer", help="Layer, Batch normalization or none."
    )
    parser.add_argument(
        '--dropout', type=float, default=0.0, help="Dropout probability."
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-4, help="Weight decay parameter."
    )
    parser.add_argument(
        '--learning_rate', type=float, default=1e-3, help="Learning rate parameter."
    )
    parser.add_argument(
        '--lr_decay', type=float, default=0.9, help="Learning rate decay parameter."
    )
    parser.add_argument(
        '--latent_loss_weight', type=float, default=1.0, help="Weight of latent MSE loss."
    )
    parser.add_argument(
        '--pixel_loss_weight', type=float, default=0.0, help="Weight of real MSE loss."
    )
    parser.add_argument(
        '--epochs', type=int, default=20, help="Number of training epochs."
    )
    parser.add_argument(
        '--batch_size', type=int, default=4, help="Number of batches."
    )
    parser.add_argument(
        '--start_date', type=str, default="2020-06-15", help="Start date."
    )
    parser.add_argument(
        '--end_date', type=str, default="2021-05-25", help="End date."
    )
    # Generation settings
    parser.add_argument(
        '--shape', type=tuple, default=(3,256,256), help='Size of the samples.')
    parser.add_argument(
        '--ckpt_dir', type=str, default ='/project/home/p200177/DE_371/resources/models/trained_generator/000024.pt')
    # Perceptual Loss
    parser.add_argument("--perceptual_loss_weight", type=float, default=1.0, help="weight of the vgg (perceptual) loss")
    parser.add_argument("--resize_input", type=float, default=0.0, help="resize input for vgg loss")
    parser.add_argument("--network_type", type=str, default='vgg16', choices=['vgg16','vgg11','vgg13','vgg19','alexnet','squeezenet1_1','resnet18','resnet34','resnet50','resnet101','resnet152','set_vit_b_16'])
    parser.add_argument("--pre_trained", action='store_true')
    parser.add_argument("--features_after_relu", action='store_true')
    parser.add_argument("--channel_computation", type=str, default='sol2', choices = ['sol1', 'sol2', 'sol3', 'sol4', 'sol5'],
                    help="Either we compute layer by layer and member per member but we have to triple the input to make it rgb or all in one (naive)")
    parser.add_argument("--network_dir", type=str, default='/project/home/p200177/DE_371/resources/network_for_perceptual_loss/', help="Insert a path")
    parser.add_argument("--style_layers", type=utils.str2intlist, default=[], help="style layers to include in vgg loss computation")
    parser.add_argument("--feature_layers", type=utils.str2intlist, default=[0,1,2,3], help="feature layers to include in vgg computation")
    parser.add_argument("--alpha_feature", type=float, default=1.0, help="weight of the feature/content loss")
    parser.add_argument("--alpha_style", type=float, default=0.01, help="weight of the style loss")
    parser.add_argument("--multi_scale_perceptual_loss",  action='store_true')

    args = parser.parse_args()
    device = args.device
    num_workers = args.num_workers
    model_name = args.model_name
    training_description = args.training_description
    weight_decay = args.weight_decay
    lr = args.learning_rate
    lr_decay = args.lr_decay
    output_shape = args.shape
    ckpt_dir = args.ckpt_dir
    inv_dir = args.inv_dir
    pack_dir = args.pack_dir
    inv_dir_val = args.inv_dir_val
    pack_dir_val = args.pack_dir_val
    epochs = args.epochs
    batch_size = args.batch_size
    start_date = args.start_date
    end_date = args.end_date

    # Set up DDP
    world_size=int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=int(os.environ['RANK']))
    device = torch.device(f"{device}:{dist.get_rank()}")  # Each process gets its GPU

    model_classes = {
        "LatentInterpolator": models.LatentInterpolator,
        "LatentInterpolatorCorrector": models.LatentInterpolatorCorrector,
        "LatentInterpolator2" : models.LatentInterpolator2,
        "LatentInterpolatorCorrector2": models.LatentInterpolatorCorrector2,
        "DualAutoencoderInterpolator": models.DualAutoencoderInterpolator,
        "DualAutoencoderInterpolatorCorrector": models.DualAutoencoderInterpolatorCorrector,
        "LatentVectorInterpolatorCorrector": models.LatentVectorInterpolatorCorrector
    }

    if dist.get_rank() == 0:
        print(args)
        print(f"Running on {world_size} {args.device} devices...")
        print(f"Using {num_workers} workers per device...")
        print(f"Model name: {model_name}")
        print(f"Loading latent space vectors from {start_date} to {end_date}...")

    training_dataset = InterpolatorDataset(
        start_date=start_date,
        end_date=end_date,
        latent_basepath=f"{inv_dir}w",
        real_basepath=f"{pack_dir}Rsemble",
        leadtimes=np.arange(1, 46, 1),
        dt=6,
        fmt='npy',
        include_input_leadtimes=True)
    
    validation_dataset = InterpolatorDataset(
        start_date=start_date,
        end_date=end_date,
        latent_basepath=f"{inv_dir_val}w",
        real_basepath=f"{pack_dir_val}Rsemble",
        leadtimes=np.arange(1, 46, 1),
        dt=6,
        fmt='npy',
        include_input_leadtimes=True)

    if dist.get_rank() == 0:
        print(f"Number of training examples: {len(training_dataset)}")
        print(f"Number of validation examples: {len(validation_dataset)}")

    # Use DistributedSampler to distribute data across multiple processes
    train_sampler = DistributedSampler(training_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    val_sampler = DistributedSampler(validation_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
    
    # Initialize model, loss function and optimizer
    if model_name in model_classes:
        model = model_classes[model_name](args).to(device)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    if dist.get_rank() == 0:
        print(f"Model architecture: {model}\n")

    # Wrap model with DDP
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])

    perceptual_loss_class = None
    if args.perceptual_loss_weight > 0:
        perceptual_loss_class = PerceptualLoss(
                                        config=args,
                                        device=device,
                                        multi_scale=args.multi_scale_perceptual_loss
                                        ).to(device).eval()
        #print("Precomputing the features...")
        #perceptual_loss_class.compute_perceptual_features(img=real_samples)

    loss_function = combined_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    if dist.get_rank() == 0:
        print("Loading the generator...\n")
    generator = load_generator(output_shape, ckpt_dir, device)

    if dist.get_rank() == 0:
        print("Starting the training...")
    for current_epoch in range(epochs):
        if dist.get_rank() == 0:
            print(f"Current learning rate: {scheduler.get_last_lr()[0]:.4e}\n")
        
        # Set the sampler epoch to ensure proper shuffling of data across epochs
        train_sampler.set_epoch(current_epoch)
        val_sampler.set_epoch(current_epoch)

        # Train and test loops
        train_loop(training_dataloader, model, generator, loss_function, optimizer, current_epoch, perceptual_loss_class, dist.get_rank(), args)
        test_loop(validation_dataloader, model, generator, loss_function, current_epoch, perceptual_loss_class, dist.get_rank(), args)
        
        scheduler.step()

        # Save model every 5 epochs
        if (current_epoch + 1) % 5 == 0:
            dt = datetime.today().strftime("%Y-%m-%dT%H_%M")
            output_name = f"/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/interpolation_models/{model_name}-{training_description}-epoch-{current_epoch+1}-{dt}.pt"
            if dist.get_rank() == 0:  # Only rank 0 saves the model
                print(f"Saving the model to {output_name}...")
                torch.save(model.state_dict(), output_name)

    if dist.get_rank() == 0:
        print("Training complete!")

    # Clean up the distributed process group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

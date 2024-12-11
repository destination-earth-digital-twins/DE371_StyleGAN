import os
import argparse
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

from gan.model.stylegan2 import Generator
from inversion.perceptual_loss.perceptual_loss import PerceptualLoss
import perturbation.utils as utils

class LatentInterpolator(nn.Module):
    def __init__(self, args, style_dims=14, latent_dims=512):
        super(LatentInterpolator, self).__init__()
        self.style_dims = style_dims
        self.latent_dims = latent_dims
        input_dim = 2 * latent_dims * style_dims + 1  # Inputs: w_start, w_end, and t

        layers = []
        for i in range(args.num_layers):
            in_features = input_dim if i == 0 else args.num_neurons
            out_features = latent_dims * style_dims if i == args.num_layers - 1 else args.num_neurons

            layers.append(nn.Linear(in_features, out_features))
            if i < args.num_layers - 1:
                if args.normalization == "Layer":
                    layers.append(nn.LayerNorm(out_features))
                elif args.normalization == "Batch":
                    layers.append(nn.BatchNorm1d(out_features))
                layers.append(nn.ReLU())
                if args.dropout > 0:
                    layers.append(nn.Dropout(p=args.dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, w_start, w_end, t):
        batch_size = w_start.size(0)  # Get batch size

        # Flatten latent space vectors
        w_start_flat = w_start.view(batch_size, -1)  # [batch_size, 14, 512] -> [batch_size, 7168]
        w_end_flat = w_end.view(batch_size, -1)      # [batch_size, 14, 512] -> [batch_size, 7168]

        # Expand `t` and concatenate inputs
        t_expanded = t.view(batch_size, 1)  # Ensure t is [batch_size, 1]
        x = torch.cat([w_start_flat, w_end_flat, t_expanded], dim=1)  # Concatenate along feature dimension

        # Pass through the feedforward network
        w_predicted = self.network(x)  # [batch_size, 7168]

        return w_predicted.view(batch_size, self.style_dims, self.latent_dims)  # [batch_size, 14, 512]

class LatentInterpolatorCorrector(nn.Module):
    def __init__(self, args, style_dims=14, latent_dims=512):
        super(LatentInterpolatorCorrector, self).__init__()
        self.style_dims = style_dims
        self.latent_dims = latent_dims
        input_dim = 2 * latent_dims * style_dims + 1  # Inputs: w_start, w_end, and t

        layers = []
        for i in range(args.num_layers):
            in_features = input_dim if i == 0 else args.num_neurons
            out_features = latent_dims * style_dims if i == args.num_layers - 1 else args.num_neurons

            layers.append(nn.Linear(in_features, out_features))
            if i < args.num_layers - 1:
                if args.normalization == "Layer":
                    layers.append(nn.LayerNorm(out_features))
                elif args.normalization == "Batch":
                    layers.append(nn.BatchNorm1d(out_features))
                layers.append(nn.ReLU())
                if args.dropout > 0:
                    layers.append(nn.Dropout(p=args.dropout)) 

        self.network = nn.Sequential(*layers)

    def forward(self, w_start, w_end, t):
        batch_size = w_start.size(0)  # Get batch size

        # Flatten latent space vectors
        w_start_flat = w_start.view(batch_size, -1)  # [batch_size, 14, 512] -> [batch_size, 7168]
        w_end_flat = w_end.view(batch_size, -1)      # [batch_size, 14, 512] -> [batch_size, 7168]

        # Expand `t` and concatenate inputs
        t_expanded = t.view(batch_size, 1)  # Ensure t is [batch_size, 1]
        x = torch.cat([w_start_flat, w_end_flat, t_expanded], dim=1)  # Concatenate along feature dimension

        # Compute linear interpolation
        w_linear_flat = w_start_flat + t_expanded * (w_end_flat - w_start_flat)  # [batch_size, 7168]
        
        # Pass through the feedforward network
        correction = self.network(x)  # [batch_size, 7168]

        # Add correction to linear interpolation
        w_corrected = w_linear_flat + correction  # [batch_size, 7168]
        
        return w_corrected.view(batch_size, self.style_dims, self.latent_dims)  # [batch_size, 14, 512]

class InterpolatorDataset(Dataset):
    def __init__(self, start_date, end_date, latent_basepath, real_basepath,
                 leadtimes=np.arange(1, 46, 1), invstep=1000, dt=6, fmt='npy'):
        self.start_date = start_date
        self.end_date = end_date
        self.latent_basepath = latent_basepath
        self.real_basepath = real_basepath
        self.leadtimes = leadtimes
        self.invstep = invstep
        self.dt = dt
        self.fmt = fmt
        self.indices = self._build_index()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Return batch corresponding to the given index."""
        date_str, t_start, t_end, t_int = self.indices[idx]
        batch = self.get_batch(date_str, t_start, t_end, t_int)
        if batch is None:
            raise RuntimeError(f"Batch for index {idx} could not be loaded.")
        return batch

    def _build_index(self):
        """Builds the list of indices for the dataset."""
        indices = []
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            for t_start in range(self.leadtimes[0], self.leadtimes[-1] - self.dt + 1):
                t_end = t_start + self.dt
                for t_int in range(1, self.dt):
                    t_int = t_start + t_int
                    # Check if the required files exist
                    latent_files_exist = all(
                        self.file_exists(self.latent_basepath, date_str, leadtime, invstep=self.invstep)
                        for leadtime in [t_start, t_int, t_end]
                    )
                    real_files_exist = all(
                        self.file_exists(self.real_basepath, date_str, leadtime)
                        for leadtime in [t_start, t_int, t_end]
                    )
                    if latent_files_exist and real_files_exist:
                        indices.append([date_str, t_start, t_end, t_int])
            current_date += timedelta(days=1)

        return indices

    def file_exists(self, basepath, date, leadtime, invstep=None):
        """Check if the required file exists."""
        if invstep:
            filename = f"{basepath}_{date}_{leadtime}_{invstep}.{self.fmt}"
        else:
            filename = f"{basepath}_{date}_{leadtime}.{self.fmt}"
        return os.path.exists(filename)

    def get_sample(self, basepath, date, leadtime, invstep=None):
        """Loads a single sample (returns None if loading fails)."""
        if invstep:
            filename = f"{basepath}_{date}_{leadtime}_{invstep}.{self.fmt}"
        else:
            filename = f"{basepath}_{date}_{leadtime}.{self.fmt}"

        if os.path.exists(filename):
            try:
                sample = np.load(filename)
                return torch.from_numpy(sample)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        return None

    def get_batch(self, date, start_leadtime, end_leadtime, int_leadtime):
        """Returns a batch of tensors (None if any part is missing)."""
        try:
            w_start = self.get_sample(self.latent_basepath, date, start_leadtime, self.invstep)
            w_t = self.get_sample(self.latent_basepath, date, int_leadtime, self.invstep)
            w_end = self.get_sample(self.latent_basepath, date, end_leadtime, self.invstep)
            r_start = self.get_sample(self.real_basepath, date, start_leadtime)
            r_t = self.get_sample(self.real_basepath, date, int_leadtime)
            r_end = self.get_sample(self.real_basepath, date, end_leadtime)

            if None in [w_start, w_t, w_end, r_start, r_t, r_end]:
                print(f"Missing data for {date} ({start_leadtime}, {end_leadtime}, {int_leadtime})")
                return None

            n_members = len(w_start)
            assert len(w_start) == len(w_t) == len(w_end) == len(r_start) == len(r_t) == len(r_end)

            t = torch.full((n_members,), (int_leadtime - start_leadtime) / self.dt, dtype=torch.float32)
            return w_start, w_end, t, w_t, r_start, r_end, r_t

        except Exception as e:
            print(f"Error creating batch for {date} ({start_leadtime}, {end_leadtime}, {int_leadtime}): {e}")
            return None

def combined_loss(w_interpolated, w_t, r_interpolated, r_t,
                        latent_loss_weight=1.0, pixel_loss_weight=0.0,
                        perceptual_loss_class=None, perceptual_loss_weight=0.0):
    latent_loss = 0.
    image_pixel_loss = 0.
    image_perceptual_loss = 0.

    if latent_loss_weight > 0:
        latent_loss = nn.MSELoss()(w_interpolated, w_t)
    if pixel_loss_weight > 0:
        image_pixel_loss = nn.MSELoss()(r_interpolated, r_t)
    if perceptual_loss_weight > 0:
        image_perceptual_loss = perceptual_loss_class(
            img_gen=r_interpolated, input_img=r_t)

    loss = latent_loss_weight * latent_loss + pixel_loss_weight * image_pixel_loss + image_perceptual_loss * perceptual_loss_weight

    return loss

def linear_interpolation(sample_start, sample_end, t):
    # Expects normalized time
    return sample_start + t * (sample_end - sample_start)

def get_mse(sample, ref, device):
    u_mse = ((ref[:, 0] - sample[:, 0])**2).mean()
    v_mse = ((ref[:, 1] - sample[:, 1])**2).mean()
    t2m_mse = ((ref[:, 2] - sample[:, 2])**2).mean()
    return torch.tensor([u_mse, v_mse, t2m_mse], device=device)

def get_mae(sample, ref, device):
    u_mse = (ref[:, 0] - sample[:, 0]).abs().mean()
    v_mse = (ref[:, 1] - sample[:, 1]).abs().mean()
    t2m_mse = (ref[:, 2] - sample[:, 2]).abs().mean()
    return torch.tensor([u_mse, v_mse, t2m_mse], device=device)

def train_loop(dataloader, model, generator, loss_function, optimizer, current_epoch, perceptual_loss_class, rank, args):
    model.train()  # Set model to train mode
    training_loss = 0.0
    num_batches = len(dataloader)

    if rank == 0:
        progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{args.epochs}", ncols=100)
    else:
        progress_bar = dataloader

    for batch in progress_bar:
        batch = [x.to(rank).view(-1, *x.shape[2:]) for x in batch]
        w_start, w_end, t, w_t, r_start, r_end, r_t = batch

        optimizer.zero_grad()

        # Forward pass
        w_interpolated = model(w_start, w_end, t)
        r_latent_nn_interpolation = generate_image_from_latent(w_interpolated, generator)

        # Compute loss
        loss = loss_function(w_interpolated, w_t, 
                             r_latent_nn_interpolation, r_t,
                             args.latent_loss_weight,
                             args.pixel_loss_weight,
                             perceptual_loss_class,
                             args.perceptual_loss_weight)

        loss.backward()
        optimizer.step()

        # Synchronize loss
        loss_tensor = torch.tensor(loss.item(), device=rank)
        dist.reduce(loss_tensor, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            global_loss = loss_tensor.item() / dist.get_world_size()
            progress_bar.set_postfix({'Training loss': f'{global_loss:.4e}'})
            training_loss += global_loss
        else:
            training_loss += loss.item()

    avg_training_loss = training_loss / num_batches
    if rank == 0:
        print(f"Epoch {current_epoch+1} - Mean training loss: {avg_training_loss:.4e}\n")

    dist.barrier()

def test_loop(dataloader, model, generator, loss_function, current_epoch, perceptual_loss_class, rank, args):
    model.eval()
    test_loss = 0.0
    
    # Initialize MSE tensors on the correct device
    phys_linear_interpolation_mse = torch.zeros(3).to(rank)
    latent_linear_interpolation_mse = torch.zeros(3).to(rank)
    latent_nn_interpolation_mse = torch.zeros(3).to(rank)
    
    num_batches = len(dataloader)

    with torch.no_grad():
        if rank == 0:
            progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{args.epochs}", ncols=100)
        else:
            progress_bar = dataloader

        for batch in progress_bar:
            batch = [x.to(rank).view(-1, *x.shape[2:]) for x in batch]
            w_start, w_end, t, w_t, r_start, r_end, r_t = batch

            # Ensure that model's output is on the correct device
            w_interpolated = model(w_start, w_end, t).to(rank)
            r_latent_nn_interpolation = generate_image_from_latent(w_interpolated, generator).to(rank)

            # Loss calculation (no need to move tensors again, they're already on the right device)
            loss = loss_function(w_interpolated, w_t,
                                 r_latent_nn_interpolation, r_t,
                                 args.latent_loss_weight,
                                 args.pixel_loss_weight,
                                 perceptual_loss_class,
                                 args.perceptual_loss_weight)

            loss_tensor = torch.tensor(loss.item(), device=rank)
            dist.reduce(loss_tensor, dst=0, op=dist.ReduceOp.SUM)
            if rank == 0:
                global_loss = loss_tensor.item() / dist.get_world_size()
                progress_bar.set_postfix({'Validation loss': f'{global_loss:.4e}'})
                test_loss += global_loss
            else:
                test_loss += loss.item()

            # Linear interpolation
            r_phys_interpolated = linear_interpolation(r_start, r_end, t[0])
            w_latent_linear_interpolation = linear_interpolation(w_start, w_end, t[0])

            # Generate latent image (ensure it's on the correct device)
            r_latent_linear_interpolation = generate_image_from_latent(
                w_latent_linear_interpolation, generator)

            # Compute MSE metrics (r_t is already on the correct device, no need to move it)
            phys_linear_interpolation_mse += get_mse(r_t, r_phys_interpolated, rank) / dist.get_world_size()
            latent_linear_interpolation_mse += get_mse(r_t, r_latent_linear_interpolation, rank) / dist.get_world_size()
            latent_nn_interpolation_mse += get_mse(r_t, r_latent_nn_interpolation, rank) / dist.get_world_size()

        # Sync all MSE metrics across ranks
        dist.reduce(phys_linear_interpolation_mse, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(latent_linear_interpolation_mse, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(latent_nn_interpolation_mse, dst=0, op=dist.ReduceOp.SUM)

        # Calculate the mean values for each metric
        mean_test_loss = test_loss / num_batches
        mean_phys_linear_interpolation_mse = phys_linear_interpolation_mse / num_batches
        mean_latent_linear_interpolation_mse = latent_linear_interpolation_mse / num_batches
        mean_latent_nn_interpolation_mse = latent_nn_interpolation_mse / num_batches

        if rank == 0:
            print(f"Epoch {current_epoch+1} - Mean validation loss: {mean_test_loss:.4e}")
            print(f"Epoch {current_epoch+1} - Physical linear interpolation MSE (1000x): {mean_phys_linear_interpolation_mse * 1E3}")
            print(f"Epoch {current_epoch+1} - Latent linear interpolation MSE (1000x): {mean_latent_linear_interpolation_mse * 1E3}")
            print(f"Epoch {current_epoch+1} - Latent NN interpolation MSE (1000x): {mean_latent_nn_interpolation_mse * 1E3}")

            relative_improvement = 100 * (mean_phys_linear_interpolation_mse - mean_latent_nn_interpolation_mse) / mean_phys_linear_interpolation_mse
            print(f"Epoch {current_epoch+1} - Relative NN interpolation improvement (compared to physical linear, %): {relative_improvement}\n")

    dist.barrier()  # Ensure synchronization between all ranks before proceeding

def load_generator(output_shape, ckpt_dir, device):
    generator = Generator(output_shape[1], 512,n_mlp=8, nb_var=output_shape[0])
    ckpt = torch.load(ckpt_dir)['g_ema']

    if 'module' in list(ckpt.items())[0][0]: #juglling with Pytorch versioning and different module packaging
        ckpt_adapt = OrderedDict()
        for k in ckpt.keys():
            k0 = k[7:]
            ckpt_adapt[k0] = ckpt[k]
        generator.load_state_dict(ckpt_adapt)
    else:
        generator.load_state_dict(ckpt)

    generator.eval()
    generator = generator.to(device)

    return generator

def generate_image_from_latent(latent_vector, g_ema, noise=None):
    if noise is not None:
        img_gen = g_ema([latent_vector], input_is_latent=True, noise=noise)
    else:
        img_gen = g_ema([latent_vector], input_is_latent=True)
    return img_gen[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device', type=str, default='cuda', help="Device to use for computation ('cpu', 'cuda', etc). Default is 'cuda'."
    )
    parser.add_argument(
        '--num_workers', type=int, default=0, help="Number of workers for the dataloader."
    )
    parser.add_argument(
        '--latent_space_vectors_path', type=str, default='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_hourly/inversion/'
    )
    parser.add_argument(
        '--real_samples_path', type=str, default='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_hourly/pack/'
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
        '--start_date', type=str, default="2021-10-01", help="Start date."
    )
    parser.add_argument(
        '--end_date', type=str, default="2021-10-03", help="End date."
    )
    parser.add_argument(
        '--training_data_ratio', type=float, default=0.9, help="Ratio of the training data to the total data."
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
    inv_dir = args.latent_space_vectors_path
    pack_dir = args.real_samples_path
    epochs = args.epochs
    batch_size = args.batch_size
    start_date = args.start_date
    end_date = args.end_date
    training_data_ratio = args.training_data_ratio

    # Set up DDP
    world_size=int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=int(os.environ['RANK']))
    device = torch.device(f"{device}:{dist.get_rank()}")  # Each process gets its GPU

    model_classes = {
        "LatentInterpolator": LatentInterpolator,
        "LatentInterpolatorCorrector": LatentInterpolatorCorrector
    }

    if dist.get_rank() == 0:
        print(args)
        print(f"Running on {world_size} {args.device} devices...")
        print(f"Using {num_workers} workers per device...")
        print(f"Model name: {model_name}")
        print(f"Loading latent space vectors from {start_date} to {end_date}...")

    dataset = InterpolatorDataset(
        start_date=start_date,
        end_date=end_date,
        latent_basepath=f"{inv_dir}w",
        real_basepath=f"{pack_dir}Rsemble",
        leadtimes=np.arange(1, 46, 1),
        dt=6,
        fmt='npy')

    indices = np.random.permutation(len(dataset))
    train_size = int(training_data_ratio * len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    if dist.get_rank() == 0:
        print(f"Number of training examples: {len(train_dataset)}")
        print(f"Number of validation examples: {len(val_dataset)}")

    # Use DistributedSampler to distribute data across multiple processes
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)

    training_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
    
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

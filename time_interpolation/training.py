from tqdm import tqdm

from collections import OrderedDict

from gan.model.stylegan2 import Generator

import torch
import torch.nn as nn
import torch.distributed as dist

def combined_loss(w_interpolated, w_t, r_interpolated=None, r_t=None,
                        latent_loss_weight=1.0, pixel_loss_weight=0.0,
                        perceptual_loss_class=None, perceptual_loss_weight=0.0):
    latent_loss = 0.
    image_pixel_loss = 0.
    image_perceptual_loss = 0.

    if latent_loss_weight > 0:
        latent_loss = nn.MSELoss()(w_interpolated, w_t)
    if pixel_loss_weight > 0:
        image_pixel_loss = nn.L1Loss()(r_interpolated, r_t)
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
        batch = [
            x.to(rank).view(-1, *x.shape[2:]) if x.dim() > 2 else x.to(rank) 
            for x in batch
        ]
        w_start, w_end, t_frac, t_encodings, w_t, r_start, r_end, r_t = batch

        optimizer.zero_grad()

        # Forward pass
        w_interpolated = model(w_start, w_end, t_frac, t_encodings)
        r_latent_nn_interpolation = None
        if args.pixel_loss_weight > 0 or args.perceptual_loss_weight > 0:
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
            batch = [
                x.to(rank).view(-1, *x.shape[2:]) if x.dim() > 2 else x.to(rank) 
                for x in batch
            ]
            w_start, w_end, t_frac, t_encodings, w_t, r_start, r_end, r_t = batch

            # Ensure that model's output is on the correct device
            w_interpolated = model(w_start, w_end, t_frac, t_encodings).to(rank)
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

            n_members = int(w_start.size(0) / t_frac.size(0))
            t_frac = torch.repeat_interleave(t_frac, repeats=n_members)

            # Linear interpolation
            r_phys_interpolated = linear_interpolation(r_start, r_end, t_frac.view(-1, 1, 1, 1))
            w_latent_linear_interpolation = linear_interpolation(w_start, w_end, t_frac.view(-1, 1, 1))

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

            ## Do not include to avoid division by zero
            #if not dataloader.dataset.include_input_leadtimes:
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

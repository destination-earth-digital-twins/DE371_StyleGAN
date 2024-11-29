import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from datetime import datetime, timedelta

class LatentInterpolator(nn.Module):
    def __init__(self, style_dims=14, latent_dims=512, hidden_neurons=512, num_layers=3):
        super(LatentInterpolator, self).__init__()
        self.style_dims = style_dims
        self.latent_dims = latent_dims
        input_dim = 2 * latent_dims * style_dims + 1  # Inputs: w_start, w_end, and t
        
        layers = []
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_neurons
            out_features = latent_dims * style_dims if i == num_layers - 1 else hidden_neurons
            
            layers.append(nn.Linear(in_features, out_features))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_features)) 
                layers.append(nn.ReLU()) 
            
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

class LatentDataset(Dataset):
    def __init__(self, latent_space_vectors, dt=6):
        self.inputs, self.ground_truth = self.build_dataset(latent_space_vectors, dt)

    def build_dataset(self, latent_space_vectors, dt):
        inputs = []
        ground_truth = []
        num_lead_times, num_ensembles, style_dims, latent_dims = latent_space_vectors.shape
        
        for t_start in range(num_lead_times - dt):
            t_end = t_start + dt
            w_start = latent_space_vectors[t_start]  # Shape: (16, 14, 512)
            w_end = latent_space_vectors[t_end]      # Shape: (16, 14, 512)
            
            for ensemble_member in range(num_ensembles):  # Iterate over each ensemble member
                for t in range(1, dt):
                    intermediate_time = t_start + t
                    w_t = latent_space_vectors[intermediate_time, ensemble_member]  # Shape: (14, 512)
                    
                    # Append input tuple and ground truth
                    inputs.append((w_start[ensemble_member], w_end[ensemble_member], t / dt))  # (14, 512), scalar
                    ground_truth.append(w_t)
                    
        return inputs, ground_truth

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        w_start, w_end, t = self.inputs[idx]  # Shapes: (14, 512), (14, 512), scalar t
        w_t = self.ground_truth[idx]          # Shape: (14, 512)
        
        # Convert to tensors
        w_start = w_start.clone().detach().float()  # Shape: (14, 512)
        w_end = w_end.clone().detach().float()      # Shape: (14, 512)
        t = torch.tensor([t], dtype=torch.float32)  # This is already correct as `t` is a scalar
        w_t = w_t.clone().detach().float()          # Shape: (14, 512)

        return w_start, w_end, t, w_t

def train_loop(dataloader, model, loss_function, optimizer, device, current_epoch, total_epochs):
    model.train()  # Set the model to training mode
    training_loss = 0.0
    num_batches = len(dataloader) 

    # Use tqdm for a progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{total_epochs}", ncols=100)

    for w_start, w_end, t, w_t, in progress_bar:
        # Move batch to device
        w_start, w_end, t, w_t = w_start.to(device), w_end.to(device), t.to(device), w_t.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        w_interpolated = model(w_start, w_end, t)
        loss = loss_function(w_interpolated, w_t)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update running loss
        training_loss += loss.item()

        # Update progress bar with loss information
        progress_bar.set_postfix({'Training loss': f'{loss.item():.4f}'})

    avg_training_loss = training_loss / num_batches
    print(f"Epoch {current_epoch+1} - Mean training loss: {avg_training_loss:.4f}")
    return avg_training_loss

def test_loop(dataloader, model, loss_function, device, current_epoch, total_epochs):
    model.eval()
    test_loss = 0.0
    num_batches = len(dataloader) 

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{total_epochs}", ncols=100)
        for w_start, w_end, t, w_t, in progress_bar:
            # Move batch to device
            w_start, w_end, t, w_t = w_start.to(device), w_end.to(device), t.to(device), w_t.to(device)

            # Forward pass
            w_interpolated = model(w_start, w_end, t)
            loss = loss_function(w_interpolated, w_t)

            # Accumulate test loss
            test_loss += loss.item()

            # Update progress bar with loss information
            progress_bar.set_postfix({'Validation loss': f'{loss.item():.4f}'})

    mean_test_loss = test_loss / num_batches
    print(f"Epoch {current_epoch+1} - Mean validation loss: {mean_test_loss:.4f}")
    return mean_test_loss

def load_samples(basename, lead_times, start_date, end_date, invstep=None, fmt="npy"):
    """
    Load latent space samples for all dates between start_date and end_date.

    Args:
        basename (str): Base path to the latent vector files.
        lead_times (list or np.array): List of lead times.
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
        invstep (int, optional): Inversion step (used in filenames).
        fmt (str): File format (default: "npy").

    Returns:
        torch.Tensor: Loaded samples as a torch tensor.
    """
    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    samples = []

    # Loop through all dates in the range
    current_date = start
    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")
        for lead_time in lead_times:
            # Construct the file path and load the sample
            if invstep:
                file_path = f"{basename}_{date_str}_{lead_time}_{invstep}.{fmt}"
            else:
                file_path = f"{basename}_{date_str}_{lead_time}.{fmt}"
            
            sample = np.load(file_path)
            samples.append(sample)
        
        # Move to the next day
        current_date += timedelta(days=1)
    
    # Convert samples to a torch tensor
    return torch.tensor(np.array(samples))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--device',
        type=str,
        default='cuda:0',
        help="Device to use for computation ('cpu', 'cuda', 'cuda:0', etc). Default is 'cuda'."
    )
    parser.add_argument(
        '-m', '--model_name',
        type=str,
        default='model',
        help="Name of the model for the output."
    )
    parser.add_argument(
        '-n', '--num_neurons',
        type=int,
        default=512,
        help="Number of hidden neurons."
    )
    parser.add_argument(
        '-l', '--num_layers',
        type=int,
        default=3,
        help="Number of hidden layers."
    )

    args = parser.parse_args()
    device = args.device
    model_name = args.model_name
    hidden_neurons = args.num_neurons
    num_layers = args.num_layers

    epochs = 25
    batch_size = 512
    start_date = "2021-10-01"
    end_date = "2021-10-28"

    print(f"Running on {device}")
    print(f"Model name: {model_name}")
    print(f"Loading latent space vectors from {start_date} to {end_date}...")
    latent_space_vectors = load_samples(
        basename="/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_october/inversion/w",
        lead_times=np.arange(1, 45, 1),
        start_date=start_date,
        end_date=end_date,
        invstep=1000
    )
    print(f"Latent space vectors dataset shape: {latent_space_vectors.shape}")

    dataset = LatentDataset(latent_space_vectors, dt=6)
    indices = np.random.permutation(len(dataset)) 

    train_size = int(0.9 * len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(val_dataset)}")

    training_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function and optimizer
    model = LatentInterpolator(hidden_neurons=hidden_neurons, num_layers=num_layers)
    print(model)
    model = model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

    for current_epoch in range(epochs):
        train_loop(training_dataloader, model, loss_function, optimizer, device, current_epoch, epochs)
        test_loop(validation_dataloader, model, loss_function, device, current_epoch, epochs)
        scheduler.step()

    dt = datetime.today().strftime("%Y-%m-%dT%H_%M")
    torch.save(
        model.state_dict(), 
        f"interpolation_models/{model_name}-epoch-{current_epoch+1}-{dt}.pt")

    print("Training complete!")

if __name__ == "__main__":
    main()
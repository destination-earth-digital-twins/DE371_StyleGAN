import torch
import torch.nn as nn

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
        t_repeats = int(batch_size / t.size(0))

        # Flatten latent space vectors
        w_start_flat = w_start.view(batch_size, -1)  # [batch_size, 14, 512] -> [batch_size, 7168]
        w_end_flat = w_end.view(batch_size, -1)      # [batch_size, 14, 512] -> [batch_size, 7168]
        t = torch.repeat_interleave(t, repeats=t_repeats).view(-1, 1)

        # Expand `t` and concatenate inputs
        x = torch.cat([w_start_flat, w_end_flat, t], dim=1)  # Concatenate along feature dimension

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
        t_repeats = int(batch_size / t.size(0))

        # Flatten latent space vectors
        w_start_flat = w_start.view(batch_size, -1)  # [batch_size, 14, 512] -> [batch_size, 7168]
        w_end_flat = w_end.view(batch_size, -1)      # [batch_size, 14, 512] -> [batch_size, 7168]
        t = torch.repeat_interleave(t, repeats=t_repeats).view(-1, 1)

        # Expand `t` and concatenate inputs
        x = torch.cat([w_start_flat, w_end_flat, t], dim=1)  # Concatenate along feature dimension

        # Compute linear interpolation
        w_linear_flat = w_start_flat + t * (w_end_flat - w_start_flat)  # [batch_size, 7168]
        
        # Pass through the feedforward network
        correction = self.network(x)  # [batch_size, 7168]

        # Add correction to linear interpolation
        w_corrected = w_linear_flat + correction  # [batch_size, 7168]
        
        return w_corrected.view(batch_size, self.style_dims, self.latent_dims)  # [batch_size, 14, 512]
    
class LatentInterpolator2(nn.Module):
    def __init__(self, args, style_dims=14, latent_dims=512):
        super(LatentInterpolator2, self).__init__()
        self.style_dims = style_dims
        self.latent_dims = latent_dims
        input_dim = 2 * latent_dims * style_dims # Inputs: w_start, w_end

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
        t_repeats = int(batch_size / t.size(0))

        # Flatten latent space vectors
        w_start_flat = w_start.view(batch_size, -1)  # [batch_size, 14, 512] -> [batch_size, 7168]
        w_end_flat = w_end.view(batch_size, -1)      # [batch_size, 14, 512] -> [batch_size, 7168]
        t = torch.repeat_interleave(t, repeats=t_repeats).view(-1, 1)

        # Expand `t` and concatenate inputs
        x = torch.cat([
            w_start_flat * (1.  - t),
            w_end_flat * t
        ], dim=1)  # Concatenate along feature dimension

        # Pass through the feedforward network
        w_predicted = self.network(x)  # [batch_size, 7168]

        return w_predicted.view(batch_size, self.style_dims, self.latent_dims)  # [batch_size, 14, 512]

class LatentInterpolatorCorrector2(nn.Module):
    def __init__(self, args, style_dims=14, latent_dims=512):
        super(LatentInterpolatorCorrector2, self).__init__()
        self.style_dims = style_dims
        self.latent_dims = latent_dims
        input_dim = 2 * latent_dims * style_dims  # Inputs: w_start, w_end

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
        t_repeats = int(batch_size / t.size(0))

        # Flatten latent space vectors
        w_start_flat = w_start.view(batch_size, -1)  # [batch_size, 14, 512] -> [batch_size, 7168]
        w_end_flat = w_end.view(batch_size, -1)      # [batch_size, 14, 512] -> [batch_size, 7168]
        t = torch.repeat_interleave(t, repeats=t_repeats).view(-1, 1)

        # Expand `t` and concatenate inputs
        x = torch.cat([
            w_start_flat * (1.  - t),
            w_end_flat * t
        ], dim=1)  # Concatenate along feature dimension

        # Compute linear interpolation
        w_linear_flat = w_start_flat + t * (w_end_flat - w_start_flat)  # [batch_size, 7168]
        
        # Pass through the feedforward network
        correction = self.network(x)  # [batch_size, 7168]

        # Add correction to linear interpolation
        w_corrected = w_linear_flat + correction  # [batch_size, 7168]
        
        return w_corrected.view(batch_size, self.style_dims, self.latent_dims)  # [batch_size, 14, 512]

class DualAutoencoderInterpolator(nn.Module):
    def __init__(self, args, style_dims=14, latent_dims=512, compressed_dim=512):
        """
        Dual autoencoder interpolator that compresses w_start and w_end to a smaller dimension, 
        interpolates them, and then reconstructs w_int.
        
        Args:
            args: Model configuration arguments.
            style_dims: Number of style dimensions (default = 14).
            latent_dims: Dimension of each style (default = 512).
            compressed_dim: Dimension of the compressed latent space (default = 512).
        """
        super(DualAutoencoderInterpolator, self).__init__()
        
        self.style_dims = style_dims
        self.latent_dims = latent_dims
        self.compressed_dim = compressed_dim

        # **Encoder network** for w_start and w_end
        self.encoder_w = self._build_encoder(args, input_dim=latent_dims * style_dims, compressed_dim=compressed_dim)

        # **Decoder network** to reconstruct w_int
        self.decoder = self._build_decoder(args, compressed_dim=compressed_dim, output_dim=latent_dims * style_dims)

    def _build_encoder(self, args, input_dim, compressed_dim):
        """Builds the encoder to reduce w_start and w_end to a lower-dimensional latent space."""
        layers = []
        for i in range(args.num_layers):
            in_features = input_dim if i == 0 else args.num_neurons
            out_features = compressed_dim if i == args.num_layers - 1 else args.num_neurons

            layers.append(nn.Linear(in_features, out_features))
            if i < args.num_layers - 1:
                if args.normalization == "Layer":
                    layers.append(nn.LayerNorm(out_features))
                elif args.normalization == "Batch":
                    layers.append(nn.BatchNorm1d(out_features))
                layers.append(nn.ReLU())
                if args.dropout > 0:
                    layers.append(nn.Dropout(p=args.dropout))
        return nn.Sequential(*layers)

    def _build_decoder(self, args, compressed_dim, output_dim):
        """Builds the decoder to reconstruct w_int from the compressed latent space."""
        layers = []
        for i in range(args.num_layers):
            in_features = compressed_dim if i == 0 else args.num_neurons
            out_features = output_dim if i == args.num_layers - 1 else args.num_neurons

            layers.append(nn.Linear(in_features, out_features))
            if i < args.num_layers - 1:
                if args.normalization == "Layer":
                    layers.append(nn.LayerNorm(out_features))
                elif args.normalization == "Batch":
                    layers.append(nn.BatchNorm1d(out_features))
                layers.append(nn.ReLU())
                if args.dropout > 0:
                    layers.append(nn.Dropout(p=args.dropout))
        return nn.Sequential(*layers)

    def forward(self, w_start, w_end, t):
        batch_size = w_start.size(0)
        t_repeats = int(batch_size / t.size(0))

        # Flatten latent space vectors
        w_start_flat = w_start.view(batch_size, -1)  # [batch_size, 14, 512] -> [batch_size, 7168]
        w_end_flat = w_end.view(batch_size, -1)      # [batch_size, 14, 512] -> [batch_size, 7168]
        t = torch.repeat_interleave(t, repeats=t_repeats).view(-1, 1)

        # Pass through encoders
        h_start = self.encoder_w(w_start_flat)  # [batch_size, 7168] -> [batch_size, 256]
        h_end = self.encoder_w(w_end_flat)      # [batch_size, 7168] -> [batch_size, 256]

        # Interpolate in latent space
        h_combined = h_start * (1 - t) + h_end * t  # Interpolation in compressed space

        # Decode to reconstruct w_int
        w_int_flat = self.decoder(h_combined)  # [batch_size, 256] -> [batch_size, 7168]
        w_int = w_int_flat.view(batch_size, self.style_dims, self.latent_dims)  # [batch_size, 7168] -> [batch_size, 14, 512]

        return w_int
    
class DualAutoencoderInterpolatorCorrector(nn.Module):
    def __init__(self, args, style_dims=14, latent_dims=512, compressed_dim=512):
        """
        Dual autoencoder interpolator that compresses w_start and w_end to a smaller dimension, 
        interpolates them, and then reconstructs w_int.
        
        Args:
            args: Model configuration arguments.
            style_dims: Number of style dimensions (default = 14).
            latent_dims: Dimension of each style (default = 512).
            compressed_dim: Dimension of the compressed latent space (default = 512).
        """
        super(DualAutoencoderInterpolatorCorrector, self).__init__()
        
        self.style_dims = style_dims
        self.latent_dims = latent_dims
        self.compressed_dim = compressed_dim

        # **Encoder network** for w_start and w_end
        self.encoder_w = self._build_encoder(args, input_dim=latent_dims * style_dims, compressed_dim=compressed_dim)

        # **Decoder network** to reconstruct w_int
        self.decoder = self._build_decoder(args, compressed_dim=compressed_dim, output_dim=latent_dims * style_dims)

    def _build_encoder(self, args, input_dim, compressed_dim):
        """Builds the encoder to reduce w_start and w_end to a lower-dimensional latent space."""
        layers = []
        for i in range(args.num_layers):
            in_features = input_dim if i == 0 else args.num_neurons
            out_features = compressed_dim if i == args.num_layers - 1 else args.num_neurons

            layers.append(nn.Linear(in_features, out_features))
            if i < args.num_layers - 1:
                if args.normalization == "Layer":
                    layers.append(nn.LayerNorm(out_features))
                elif args.normalization == "Batch":
                    layers.append(nn.BatchNorm1d(out_features))
                layers.append(nn.ReLU())
                if args.dropout > 0:
                    layers.append(nn.Dropout(p=args.dropout))
        return nn.Sequential(*layers)

    def _build_decoder(self, args, compressed_dim, output_dim):
        """Builds the decoder to reconstruct w_int from the compressed latent space."""
        layers = []
        for i in range(args.num_layers):
            in_features = compressed_dim if i == 0 else args.num_neurons
            out_features = output_dim if i == args.num_layers - 1 else args.num_neurons

            layers.append(nn.Linear(in_features, out_features))
            if i < args.num_layers - 1:
                if args.normalization == "Layer":
                    layers.append(nn.LayerNorm(out_features))
                elif args.normalization == "Batch":
                    layers.append(nn.BatchNorm1d(out_features))
                layers.append(nn.ReLU())
                if args.dropout > 0:
                    layers.append(nn.Dropout(p=args.dropout))
        return nn.Sequential(*layers)

    def forward(self, w_start, w_end, t):
        batch_size = w_start.size(0)
        t_repeats = int(batch_size / t.size(0))

        # Flatten latent space vectors
        w_start_flat = w_start.view(batch_size, -1)  # [batch_size, 14, 512] -> [batch_size, 7168]
        w_end_flat = w_end.view(batch_size, -1)      # [batch_size, 14, 512] -> [batch_size, 7168]
        t = torch.repeat_interleave(t, repeats=t_repeats).view(-1, 1)

        # Pass through encoders
        h_start = self.encoder_w(w_start_flat)  # [batch_size, 7168] -> [batch_size, 256]
        h_end = self.encoder_w(w_end_flat)      # [batch_size, 7168] -> [batch_size, 256]

        # Interpolate in latent space
        h_combined = h_start * (1 - t) + h_end * t  # Interpolation in compressed space

        # Decode to reconstruct w_int
        correction = self.decoder(h_combined)  # [batch_size, 256] -> [batch_size, 7168]

        # Compute linear interpolation
        w_linear_flat = w_start_flat + t * (w_end_flat - w_start_flat)  # [batch_size, 7168]

        # Add correction to linear interpolation
        w_corrected = w_linear_flat + correction  # [batch_size, 7168]
        
        return w_corrected.view(batch_size, self.style_dims, self.latent_dims)  # [batch_size, 14, 512]

class LatentVectorInterpolator(nn.Module):
    def __init__(self, args, style_dims=14, latent_dims=512):
        super(LatentVectorInterpolator, self).__init__()
        self.style_dims = style_dims
        self.latent_dims = latent_dims
        self.num_neurons = args.num_neurons
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        input_dim = 2 * latent_dims  # Inputs: w_start, w_end

        layers = []
        for i in range(args.num_layers):
            in_features = input_dim if i == 0 else self.num_neurons
            out_features = latent_dims if i == self.num_layers - 1 else self.num_neurons

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
        batch_size, num_channels, feature_dim = w_start.size()  # (batch_size, 14, 512)
        t_repeats = int(batch_size * num_channels / t.size(0))
        
        w_start = w_start.view(batch_size * num_channels, feature_dim)  # Shape: (batch_size * 14, 512)
        w_end = w_end.view(batch_size * num_channels, feature_dim)  # Shape: (batch_size * 14, 512)
        t = torch.repeat_interleave(t, repeats=t_repeats).view(-1, 1)

        # Expand `t` and concatenate inputs
        x = torch.cat([
            w_start * (1.  - t),
            w_end * t
        ], dim=1)  # Concatenate along feature dimension
        
        # Pass through the feedforward network
        w_int = self.network(x)  # [batch_size, 512]
        w_int = w_int.view(batch_size, num_channels, -1)  # Shape: (batch_size, 14, 512)
        
        return w_int

class LatentVectorInterpolatorCorrector(nn.Module):
    def __init__(self, args, style_dims=14, latent_dims=512):
        super(LatentVectorInterpolatorCorrector, self).__init__()
        self.style_dims = style_dims
        self.latent_dims = latent_dims
        self.num_neurons = args.num_neurons
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        input_dim = 2 * latent_dims  # Inputs: w_start, w_end

        layers = []
        for i in range(args.num_layers):
            in_features = input_dim if i == 0 else self.num_neurons
            out_features = latent_dims if i == self.num_layers - 1 else self.num_neurons

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
        batch_size, num_channels, feature_dim = w_start.size()  # (batch_size, 14, 512)
        t_repeats = int(batch_size * num_channels / t.size(0))
        
        w_start = w_start.view(batch_size * num_channels, feature_dim)  # Shape: (batch_size * 14, 512)
        w_end = w_end.view(batch_size * num_channels, feature_dim)  # Shape: (batch_size * 14, 512)
        t = torch.repeat_interleave(t, repeats=t_repeats).view(-1, 1)

        # Expand `t` and concatenate inputs
        x = torch.cat([
            w_start * (1.  - t),
            w_end * t
        ], dim=1)  # Concatenate along feature dimension

        # Compute linear interpolation
        w_linear = w_start + t * (w_end - w_start)  # [batch_size, 512]
        
        # Pass through the feedforward network
        correction = self.network(x)  # [batch_size, 512]

        # Add correction to linear interpolation
        w_corrected = w_linear + correction
        w_corrected = w_corrected.view(batch_size, num_channels, -1)  # Shape: (batch_size, 14, 512)
        
        return w_corrected
    
class LatentVectorInterpolatorCorrector2(nn.Module):
    def __init__(self, args, style_dims=14, latent_dims=512):
        super(LatentVectorInterpolatorCorrector2, self).__init__()
        self.style_dims = style_dims
        self.latent_dims = latent_dims

        # Create a separate network for each style dimension
        self.sub_networks = nn.ModuleList()
        for _ in range(style_dims):
            layers = []
            input_dim = 2 * latent_dims  # Inputs: w_start, w_end
            for i in range(args.num_layers):
                in_features = input_dim if i == 0 else args.num_neurons
                out_features = latent_dims if i == args.num_layers - 1 else args.num_neurons

                layers.append(nn.Linear(in_features, out_features))
                if i < args.num_layers - 1:
                    if args.normalization == "Layer":
                        layers.append(nn.LayerNorm(out_features))
                    elif args.normalization == "Batch":
                        layers.append(nn.BatchNorm1d(out_features))
                    layers.append(nn.ReLU())
                    if args.dropout > 0:
                        layers.append(nn.Dropout(p=args.dropout))

            self.sub_networks.append(nn.Sequential(*layers))

    def forward(self, w_start, w_end, t):
        batch_size, num_channels, feature_dim = w_start.size()  # (batch_size, 14, 512)
        assert num_channels == self.style_dims, "Mismatch in number of style dimensions"

        # Repeat `t` to match the batch size and number of channels
        t_repeats = int(batch_size * num_channels / t.size(0))
        t = torch.repeat_interleave(t, repeats=t_repeats).view(batch_size, num_channels, 1)

        # Compute linear interpolation
        w_linear = w_start + t * (w_end - w_start)  # Shape: (batch_size, 14, 512)

        # Process each style dimension independently
        corrections = []
        for i in range(self.style_dims):
            # Extract the i-th style vector across the batch
            w_start_i = w_start[:, i, :]  # Shape: (batch_size, 512)
            w_end_i = w_end[:, i, :]  # Shape: (batch_size, 512)
            t_i = t[:, i, :]  # Shape: (batch_size, 1)

            # Concatenate inputs for the i-th style dimension
            x_i = torch.cat([w_start_i * (1 - t_i), w_end_i * t_i], dim=1)  # Shape: (batch_size, 1024)

            # Pass through the i-th sub-network
            correction_i = self.sub_networks[i](x_i)  # Shape: (batch_size, 512)
            corrections.append(correction_i.unsqueeze(1))  # Add channel dimension

        # Combine corrections for all style dimensions
        corrections = torch.cat(corrections, dim=1)  # Shape: (batch_size, 14, 512)

        # Add corrections to the linear interpolation
        w_corrected = w_linear + corrections  # Shape: (batch_size, 14, 512)

        return w_corrected
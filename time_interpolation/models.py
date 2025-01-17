import torch
import torch.nn as nn

class LatentCodeInterpolator(nn.Module):
    def __init__(self, args, style_dims=14, latent_dims=512, time_frac_dims=1, time_encoding_dims=12):
        super(LatentCodeInterpolator, self).__init__()
        self.style_dims = style_dims
        self.latent_dims = latent_dims
        self.time_frac_dims = time_frac_dims
        self.time_encoding_dims = time_encoding_dims
        input_dim = 2 * latent_dims * style_dims + time_frac_dims + time_encoding_dims

        layers = []
        for i in range(args.num_layers):
            in_features = input_dim if i == 0 else args.num_neurons
            out_features = latent_dims * style_dims if i == args.num_layers - 1 else args.num_neurons

            layers.append(nn.Linear(in_features, out_features))
            if i < args.num_layers - 1:
                if args.normalization == "Layer":
                    layers.append(nn.LayerNorm(out_features))
                layers.append(nn.ReLU())
                if args.dropout > 0:
                    layers.append(nn.Dropout(p=args.dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, w_start, w_end, t_frac, t_encodings):
        batch_size = w_start.size(0)
        t_repeats = int(batch_size / t_frac.size(0)) # Gives the number of ensemble members
        
        # Flatten latent space vectors
        w_start_flat = w_start.view(batch_size, -1)  # [batch_size, 14, 512] -> [batch_size, 7168]
        w_end_flat = w_end.view(batch_size, -1)      # [batch_size, 14, 512] -> [batch_size, 7168]
        t_frac = torch.repeat_interleave(t_frac, repeats=t_repeats, dim=0).view(-1, self.time_frac_dims)
        t_encodings = torch.repeat_interleave(t_encodings, repeats=t_repeats, dim=0).view(-1, self.time_encoding_dims)

        # Expand `t` and concatenate inputs
        x = torch.cat([
            w_start_flat,
            w_end_flat,
            t_frac,
            t_encodings
        ], dim=1)  # Concatenate along feature dimension
        # Pass through the feedforward network
        w_predicted = self.network(x)  # [batch_size, 7168]

        return w_predicted.view(batch_size, self.style_dims, self.latent_dims)  # [batch_size, 14, 512]

class LatentCodeInterpolatorCorrector(LatentCodeInterpolator):
    def forward(self, w_start, w_end, t_frac, t_encodings):
        batch_size = w_start.size(0)
        t_repeats = int(batch_size / t_frac.size(0)) # Gives the number of ensemble members

        # Flatten latent space vectors
        w_start_flat = w_start.view(batch_size, -1)  # [batch_size, 14, 512] -> [batch_size, 7168]
        w_end_flat = w_end.view(batch_size, -1)      # [batch_size, 14, 512] -> [batch_size, 7168]
        t_frac = torch.repeat_interleave(t_frac, repeats=t_repeats, dim=0).view(-1, self.time_frac_dims)
        t_encodings = torch.repeat_interleave(t_encodings, repeats=t_repeats, dim=0).view(-1, self.time_encoding_dims)

        # Expand `t` and concatenate inputs
        x = torch.cat([
            w_start_flat,
            w_end_flat,
            t_frac,
            t_encodings
        ], dim=1)  # Concatenate along feature dimension

        # Compute linear interpolation
        w_linear_flat = w_start_flat + t_frac * (w_end_flat - w_start_flat)  # [batch_size, 7168]
        
        # Pass through the feedforward network
        correction = self.network(x)  # [batch_size, 7168]

        # Add correction to linear interpolation
        w_corrected = w_linear_flat + correction  # [batch_size, 7168]
        
        return w_corrected.view(batch_size, self.style_dims, self.latent_dims)  # [batch_size, 14, 512]

class StyleVectorInterpolator(nn.Module):
    def __init__(self, args, style_dims=14, latent_dims=512, time_frac_dims=1, time_encoding_dims=12):
        super(StyleVectorInterpolator, self).__init__()
        self.style_dims = style_dims
        self.latent_dims = latent_dims
        self.time_frac_dims = time_frac_dims
        self.time_encoding_dims = time_encoding_dims

        # Create a separate network for each style dimension
        self.sub_networks = nn.ModuleList()
        for _ in range(style_dims):
            layers = []
            input_dim = 2 * latent_dims + time_frac_dims + time_encoding_dims
            for i in range(args.num_layers):
                in_features = input_dim if i == 0 else args.num_neurons
                out_features = latent_dims if i == args.num_layers - 1 else args.num_neurons

                layers.append(nn.Linear(in_features, out_features))
                if i < args.num_layers - 1:
                    if args.normalization == "Layer":
                        layers.append(nn.LayerNorm(out_features))
                    layers.append(nn.ReLU())
                    if args.dropout > 0:
                        layers.append(nn.Dropout(p=args.dropout))

            self.sub_networks.append(nn.Sequential(*layers))

    def forward(self, w_start, w_end, t_frac, t_encodings):
        batch_size = w_start.size(0)
        t_repeats = int(batch_size / t_frac.size(0)) # Gives the number of ensemble members
        t_frac = torch.repeat_interleave(t_frac, repeats=t_repeats, dim=0).view(-1, self.time_frac_dims)
        t_encodings = torch.repeat_interleave(t_encodings, repeats=t_repeats, dim=0).view(-1, self.time_encoding_dims)
        
        # Process each style dimension independently
        predictions = []
        for i in range(self.style_dims):
            # Extract the i-th style vector across the batch
            w_start_i = w_start[:, i, :]  # Shape: (batch_size, 512)
            w_end_i = w_end[:, i, :]  # Shape: (batch_size, 512)

            # Concatenate inputs for the i-th style dimension
            x_i = torch.cat([
                w_start_i, 
                w_end_i,
                t_frac,
                t_encodings,
            ], dim=1)

            # Pass through the i-th sub-network
            prediction_i = self.sub_networks[i](x_i)  # Shape: (batch_size, 512)
            predictions.append(prediction_i.unsqueeze(1))  # Add channel dimension

        # Combine predictions for all style dimensions
        predictions = torch.cat(predictions, dim=1)  # Shape: (batch_size, 14, 512)

        return predictions
        
class StyleVectorInterpolatorCorrector(StyleVectorInterpolator):
    def forward(self, w_start, w_end, t_frac, t_encodings):
        batch_size = w_start.size(0)
        t_repeats = int(batch_size / t_frac.size(0)) # Gives the number of ensemble members
        t_frac = torch.repeat_interleave(t_frac, repeats=t_repeats, dim=0).view(-1, self.time_frac_dims)
        t_encodings = torch.repeat_interleave(t_encodings, repeats=t_repeats, dim=0).view(-1, self.time_encoding_dims)
        w_start_flat = w_start.view(batch_size, -1)  # [batch_size, 14, 512] -> [batch_size, 7168]
        w_end_flat = w_end.view(batch_size, -1)      # [batch_size, 14, 512] -> [batch_size, 7168]
        w_linear = w_start_flat + t_frac * (w_end_flat - w_start_flat)  # Shape: (batch_size, 14, 512)

        # Process each style dimension independently
        corrections = []
        for i in range(self.style_dims):
            # Extract the i-th style vector across the batch
            w_start_i = w_start[:, i, :]  # Shape: (batch_size, 512)
            w_end_i = w_end[:, i, :]  # Shape: (batch_size, 512)

            # Concatenate inputs for the i-th style dimension
            x_i = torch.cat([
                w_start_i,
                w_end_i,
                t_frac,
                t_encodings,
            ], dim=1)

            # Pass through the i-th sub-network
            correction_i = self.sub_networks[i](x_i)  # Shape: (batch_size, 512)
            corrections.append(correction_i.unsqueeze(1))  # Add channel dimension

        # Combine corrections for all style dimensions
        corrections = torch.cat(corrections, dim=1)  # Shape: (batch_size, 14, 512)

        # Add corrections to the linear interpolation
        w_corrected = w_linear.view(batch_size, self.style_dims, -1) + corrections  # Shape: (batch_size, 14, 512)

        return w_corrected

class StyleVectorInterpolatorExtended(nn.Module):
    def __init__(self, args, style_dims=14, latent_dims=512, time_frac_dims=1, time_encoding_dims=12):
        super(StyleVectorInterpolatorExtended, self).__init__()
        self.style_dims = style_dims
        self.latent_dims = latent_dims
        self.time_frac_dims = time_frac_dims
        self.time_encoding_dims = time_encoding_dims

        # Create a separate network for each style dimension
        self.sub_networks = nn.ModuleList()
        for _ in range(style_dims):
            layers = []
            input_dim = 2 * latent_dims + time_frac_dims + time_encoding_dims
            for i in range(args.num_layers):
                in_features = input_dim if i == 0 else args.num_neurons
                out_features = latent_dims if i == args.num_layers - 1 else args.num_neurons

                layers.append(nn.Linear(in_features, out_features))
                if i < args.num_layers - 1:
                    if args.normalization == "Layer":
                        layers.append(nn.LayerNorm(out_features))
                    layers.append(nn.ReLU())
                    if args.dropout > 0:
                        layers.append(nn.Dropout(p=args.dropout))

            self.sub_networks.append(nn.Sequential(*layers))
            
        layers = []
        num_final_layers = 2
        for i in range(num_final_layers):
            features = self.style_dims * self.latent_dims 
            layers.append(nn.Linear(features, features))
            if i < num_final_layers - 1:
                if args.normalization == "Layer":
                    layers.append(nn.LayerNorm(features))
                layers.append(nn.ReLU())
                if args.dropout > 0:
                    layers.append(nn.Dropout(p=args.dropout))

        self.final_network = nn.Sequential(*layers)
        
    def forward(self, w_start, w_end, t_frac, t_encodings):
        batch_size = w_start.size(0)
        t_repeats = int(batch_size / t_frac.size(0)) # Gives the number of ensemble members
        t_frac = torch.repeat_interleave(t_frac, repeats=t_repeats, dim=0).view(-1, self.time_frac_dims)
        t_encodings = torch.repeat_interleave(t_encodings, repeats=t_repeats, dim=0).view(-1, self.time_encoding_dims)
        
        # Process each style dimension independently
        predictions = []
        for i in range(self.style_dims):
            # Extract the i-th style vector across the batch
            w_start_i = w_start[:, i, :]  # Shape: (batch_size, 512)
            w_end_i = w_end[:, i, :]  # Shape: (batch_size, 512)

            # Concatenate inputs for the i-th style dimension
            x_i = torch.cat([
                w_start_i, 
                w_end_i,
                t_frac,
                t_encodings,
            ], dim=1)

            # Pass through the i-th sub-network
            prediction_i = self.sub_networks[i](x_i)  # Shape: (batch_size, 512)
            predictions.append(prediction_i.unsqueeze(1))  # Add channel dimension

        # Combine predictions for all style dimensions
        predictions = torch.cat(predictions, dim=1)  # Shape: (batch_size, 14, 512)

        final_predictions = self.final_network(predictions.view(batch_size, -1))

        return final_predictions.view(batch_size, self.style_dims, self.latent_dims)
        
class StyleVectorInterpolatorCorrectorExtended(StyleVectorInterpolatorExtended):
    def forward(self, w_start, w_end, t_frac, t_encodings):
        batch_size = w_start.size(0)
        t_repeats = int(batch_size / t_frac.size(0)) # Gives the number of ensemble members
        t_frac = torch.repeat_interleave(t_frac, repeats=t_repeats, dim=0).view(-1, self.time_frac_dims)
        t_encodings = torch.repeat_interleave(t_encodings, repeats=t_repeats, dim=0).view(-1, self.time_encoding_dims)
        w_start_flat = w_start.view(batch_size, -1)  # [batch_size, 14, 512] -> [batch_size, 7168]
        w_end_flat = w_end.view(batch_size, -1)      # [batch_size, 14, 512] -> [batch_size, 7168]
        w_linear = w_start_flat + t_frac * (w_end_flat - w_start_flat)  # Shape: (batch_size, 14, 512)

        # Process each style dimension independently
        corrections = []
        for i in range(self.style_dims):
            # Extract the i-th style vector across the batch
            w_start_i = w_start[:, i, :]  # Shape: (batch_size, 512)
            w_end_i = w_end[:, i, :]  # Shape: (batch_size, 512)

            # Concatenate inputs for the i-th style dimension
            x_i = torch.cat([
                w_start_i, 
                w_end_i,
                t_frac,
                t_encodings,
            ], dim=1)

            # Pass through the i-th sub-network
            correction_i = self.sub_networks[i](x_i)  # Shape: (batch_size, 512)
            corrections.append(correction_i.unsqueeze(1))  # Add channel dimension

        # Combine corrections for all style dimensions
        corrections = torch.cat(corrections, dim=1)  # Shape: (batch_size, 14, 512)

        final_corrections = self.final_network(corrections.view(batch_size, -1))

        # Add corrections to the linear interpolation
        w_corrected = w_linear.view(batch_size, self.style_dims, -1) + \
            final_corrections.view(batch_size, self.style_dims, -1)  # Shape: (batch_size, 14, 512)

        return w_corrected
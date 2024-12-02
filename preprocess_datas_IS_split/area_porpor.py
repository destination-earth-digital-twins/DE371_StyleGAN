import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths to adjust
chemin_csv = 'path to labels.csv'  # The path to your CSV file
dossier_samples = 'path to dataset '  # The folder containing the sample files

# Read the CSV file containing the sample names
samples_csv = pd.read_csv(chemin_csv)

# List to store all the sample data
all_samples_data = []

# Loop through each sample name in the CSV
for sample_name in samples_csv['Name']:
    # Build the path to the data file for this sample
    chemin_sample = os.path.join(dossier_samples, f'{sample_name}.npy')
    # Read the sample data (assuming each sample file is a .npy file)
    try:
        data_sample = np.load(chemin_sample)
        # If the file contains multiple columns, you can either select one or take an average
        # For this example, we use the first column
        all_samples_data.append(data_sample[0])  # We add the first column of the sample data
        print(data_sample[0].shape)
    except FileNotFoundError:
        print(f"The file for the sample {sample_name} is not found.")
        continue

# Combine all the sample data into a single DataFrame
combined_data = pd.concat(all_samples_data, axis=1)

# Calculate the minimum and maximum value of the data
val_min = combined_data.min().min()
val_max = combined_data.max().max()

# Generate a series of thresholds between the minimum and maximum values
seuils = np.linspace(val_min, val_max, 100)

# Calculate the proportion of samples > each threshold
proportions = [(combined_data > seuil).mean().mean() for seuil in seuils]

# Graph configuration
plt.figure(figsize=(10, 6))
plt.plot(seuils, proportions, marker='o')
plt.title("Proportion of samples exceeding a threshold")
plt.xlabel("Threshold")
plt.ylabel("Proportion of samples > threshold")
plt.grid(True)

# Save the graph to a file (e.g., "graphique.png")
plt.savefig("graphique.png")


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Chemins à adapter
chemin_csv = '/project/home/p200177/DE_371/datasets/dataset_Meteo_France_rr_u_v_t2m/datasaved5_0.001_500_prem/INST1/labels.csv'  # Le chemin vers ton fichier CSV
dossier_samples = '/project/home/p200177/DE_371/datasets/dataset_Meteo_France_rr_u_v_t2m/data/IS_rr_debug_1_1.0_0_0_0_0_0_256_large_lt/'  # Le dossier contenant les fichiers des samples

# Lire le fichier CSV contenant les noms des samples
samples_csv = pd.read_csv(chemin_csv)

# Liste pour stocker toutes les données des samples
all_samples_data = []

# Parcourir chaque nom de sample dans le CSV
for sample_name in samples_csv['Name']:
    # Construire le chemin vers le fichier de données pour ce sample
    chemin_sample = os.path.join(dossier_samples, f'{sample_name}.npy')
    # Lire les données du sample (on suppose que chaque fichier sample est un CSV)
    try:
        data_sample = np.load(chemin_sample)
        print('SHAPE SAMPLE', data_sample.shape)
        # Si le fichier contient plusieurs colonnes, on peut soit les sélectionner, soit prendre une moyenne
        # Pour l'exemple, on utilise la première colonne
        all_samples_data.append(data_sample[0])  # On ajoute la première colonne de données du sample
        print(data_sample[0].shape)
    except FileNotFoundError:
        print(f"Le fichier pour le sample {sample_name} est introuvable.")
        continue

# Combiner toutes les données des samples en un seul DataFrame
combined_data = pd.concat(all_samples_data, axis=1)

# Calculer la valeur minimale et maximale des données
val_min = combined_data.min().min()
val_max = combined_data.max().max()

# Générer une série de seuils entre la valeur minimale et maximale
seuils = np.linspace(val_min, val_max, 100)

# Calculer la proportion des samples > chaque seuil
proportions = [(combined_data > seuil).mean().mean() for seuil in seuils]



# Configuration du graphique
plt.figure(figsize=(10, 6))
plt.plot(seuils, proportions, marker='o')
plt.title("Proportion des samples dépassant un seuil")
plt.xlabel("Seuil")
plt.ylabel("Proportion des samples > seuil")
plt.grid(True)

# Enregistrer le graphique dans un fichier (par exemple "graphique.png")
plt.savefig("graphique.png")

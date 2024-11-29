import numpy as np
import os

# Chemin vers le dossier contenant vos fichiers .npy
dossier = '/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/VGG/rdm/sol2/amse/scores/all_pack'

# Liste pour stocker les données 'rr' de tous les fichiers
toutes_donnees_rr = []

# Fonction pour extraire les données 'rr' d'un fichier
def extraire_rr(fichier):
    # Charger le fichier .npy
    donnees = np.load(fichier, allow_pickle=True).astype(np.float32)[:,np.newaxis,:,:]  # Suppose que le fichier contient un dictionnaire
    
    # Extraire la variable 'rr'
    
    return np.exp((donnees[0][0][0]+1)*5.78319931/2)-1
    # return donnees[0][0][0]

    # else:
    #     print(f"La variable 'rr' n'existe pas dans le fichier {fichier}")
    #     return None

# Lister tous les fichiers .npy dans le dossier
fichiers_npy = [f for f in os.listdir(dossier) if f.endswith('.npy')]

# Parcourir tous les fichiers et collecter les données 'rr'
for fichier in fichiers_npy:
    chemin_complet = os.path.join(dossier, fichier)
    rr = extraire_rr(chemin_complet)
    
    if rr is not None:
        # Ajouter les données 'rr' extraites à la liste
        toutes_donnees_rr.extend(rr)  # Utiliser .extend pour concaténer les valeurs

# Convertir la liste en un tableau NumPy
toutes_donnees_rr = np.array(toutes_donnees_rr)

# Calculer le 90e percentile (Q90) sur toutes les données 'rr' réunies
q90 = np.percentile(toutes_donnees_rr, 1)

# Afficher le résultat
print(f"Le 90e percentile (Q90) de toutes les données 'rr' réunies est : {q90}")
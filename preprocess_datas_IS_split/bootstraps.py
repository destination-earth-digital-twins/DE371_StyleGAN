import pandas as pd
import glob

"Permet de prendre n csv files, et retourne un csv file sans duplicats"
# Chemin où sont stockés les fichiers CSV (remplacez 'path/to/csv_files/' par votre chemin réel)
chemin_dossier = '/project/home/p200177/DE_371/datasets/dataset_Meteo_France_rr_u_v_t2m/bootstraps/*.csv'

# Liste pour stocker les DataFrames de chaque fichier CSV
dataframes = []

# Charger tous les fichiers CSV et les ajouter à la liste `dataframes`
for fichier in glob.glob(chemin_dossier):
    print(fichier)
    df = pd.read_csv(fichier)
    dataframes.append(df)

# Combiner tous les DataFrames en un seul
df_combined = pd.concat(dataframes, ignore_index=True)

# Supprimer les doublons basés sur la colonne 'Name'
df_unique = df_combined.drop_duplicates(subset=['Name'])

# Sauvegarder le DataFrame final dans un fichier CSV
df_unique.to_csv('IS_boostrap_rr_cumul_correct.csv', index=False)

print("Le fichier CSV final sans doublons a été créé avec succès.")

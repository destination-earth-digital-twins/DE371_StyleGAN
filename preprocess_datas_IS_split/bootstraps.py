import pandas as pd
import glob

#Allows to take n csv files, and returns a csv file without duplicates.
# Path where CSV files are stored (replace 'path/to/csv_files/' with your actual path)
chemin_dossier = '/*.csv'

# List to store DataFrames for each CSV file
dataframes = []

# Load all CSV files and add them to the `dataframes` list
for fichier in glob.glob(chemin_dossier):
    print(fichier)
    df = pd.read_csv(fichier)
    dataframes.append(df)

# Combien all DF into one
df_combined = pd.concat(dataframes, ignore_index=True)

# Delete duplicates based on the 'Name' column
df_unique = df_combined.drop_duplicates(subset=['Name'])

# Save the final DataFrame as a CSV file
df_unique.to_csv('name_of_new_csv_file', index=False)

print("The final CSV file without duplicates has been successfully created.")
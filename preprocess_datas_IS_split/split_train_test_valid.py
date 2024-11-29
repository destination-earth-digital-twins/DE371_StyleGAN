import pandas as pd


  "This file allows you to split the dataset into train/test/validation. It takes a csv with dates. 
  "It returns --> the training set in the defined interval, the same for the test set and takes one week per month for the validation set.”
  
  def split_csv_with_validation_first(input_file, train_file, test_file, valid_file, train_start_date, train_end_date, test_start_date, test_end_date):
    # Lire le fichier CSV
    df = pd.read_csv(input_file)
    
    # Convertir la colonne 'DATE' en format datetime
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    print(df['Date'])
    # Supprimer les lignes avec des dates invalides
    # df0 = df.dropna(subset=['Date'])
    
    # df_extract = df0[
    #         (df0['Date'] >= train_start_date) & (df0['Date'] < train_end_date)]
    # print(df_extract)
    
    # Extraire une semaine par mois pour le jeu de validation
    validation_data = df.groupby(df['Date'].dt.to_period("M")).apply(
        lambda x: x[x['Date'].dt.isocalendar().week == x['Date'].dt.isocalendar().week.iloc[0]]
    ).reset_index(drop=True)
    validation_data['Date'] = validation_data['Date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # # Créer un ensemble de dates utilisées pour le jeu de validation pour éviter les doublons

    validation_dates = set(validation_data['Date'])
    
    # # Filtrer les données restantes en excluant les dates utilisées pour la validation

    remaining_data = df[~df['Date'].isin(validation_dates)]

    remaining_data['Date'] = remaining_data['Date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    
    
    # Filtrer les données pour le train et le test en fonction des intervalles de dates
    train_data = remaining_data[(remaining_data['Date'] >= train_start_date) & (remaining_data['Date'] <= train_end_date)]
    test_data = remaining_data[(remaining_data['Date'] >= test_start_date) & (remaining_data['Date'] <= test_end_date)]
    
    # Sauvegarder les résultats dans trois fichiers CSV distincts
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    validation_data.to_csv(valid_file, index=False)
    
    print(f"Validation data saved to {valid_file} with {len(validation_data)} records.")
    print(f"Train data saved to {train_file} with {len(train_data)} records.")
    print(f"Test data saved to {test_file} with {len(test_data)} records.")

# Exemple d'utilisation
input_file = "/home/users/u101957/DE371_StyleGAN/importance_sampling/IS_boostrap_no_duplicate_rr_cumul_correct.csv"           # Nom du fichier CSV d'entrée
train_file = "IS_boostrap_no_duplicate_rr_cumul_correct_train.csv"     # Nom du fichier CSV de sortie pour le train
test_file = "IS_boostrap_no_duplicate_rr_cumul_correct_test.csv"       # Nom du fichier CSV de sortie pour le test
valid_file = "IS_boostrap_no_duplicate_rr_cumul_correct_valid.csv"     # Nom du fichier CSV de sortie pour la validation

# Définir les intervalles de dates pour le train et le test
train_start_date = "2020-06-15"
train_end_date = "2021-06-01"
test_start_date = "2021-06-01"
test_end_date = "2021-11-12"

# Appel de la fonction
split_csv_with_validation_first(input_file, train_file, test_file, valid_file, train_start_date, train_end_date, test_start_date, test_end_date)

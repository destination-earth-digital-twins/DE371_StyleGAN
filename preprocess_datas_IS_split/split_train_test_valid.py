import pandas as pd


def split_csv_with_validation_first(params,train_start_date,test_end_date,test_start_date):
  """This function split the original csv file in valid/train/split dataframe
  """
  
  df = pd.read_csv(params.output_csv)
  df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Extract one week per month for validation 
  validation_data = df.groupby(df['Date'].dt.to_period("M")).apply(
      lambda x: x[x['Date'].dt.isocalendar().week == x['Date'].dt.isocalendar().week.iloc[0]]
  ).reset_index(drop=True)
  validation_data['Date'] = validation_data['Date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
  validation_dates = set(validation_data['Date'])
    
    #filter validation csv and test and train

  remaining_data = df[~df['Date'].isin(validation_dates)]
  remaining_data['Date'] = remaining_data['Date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

  train_data = remaining_data[(remaining_data['Date'] >= train_start_date) & (remaining_data['Date'] <= train_end_date)]
  test_data = remaining_data[(remaining_data['Date'] >= test_start_date) & (remaining_data['Date'] <= test_end_date)]
    
  # Sauvegarder les résultats dans trois fichiers CSV distincts
  train_data.to_csv(f"{params.load_directory}/IS_bootstrap_no_duplicate_rr_cumul_correct_train.csv", index=False)
  test_data.to_csv(f"{params.load_directory}/IS_bootstrap_no_duplicate_rr_cumul_correct_test.csv", index=False)
  validation_data.to_csv(f"{params.load_directory}/IS_bootstrap_no_duplicate_rr_cumul_correct_valid.csv", index=False)
    
  print(f"Validation data saved to {valid_file} with {len(validation_data)} records.")
  print(f"Train data saved to {train_file} with {len(train_data)} records.")
  print(f"Test data saved to {test_file} with {len(test_data)} records.")
  
def More_than(n,params):
  
  IS= pd.read_csv(f"{params.main_path}{params.giga_directory}{params.output_csv}")
  print(f"{params.main_path}{params.giga_directory}{params.output_csv}")
  labels = pd.read_csv(f"{params.main_path}/{params.giga_directory}/labels.csv")
  # Group data by leadtime and date 
  grouped_data = IS.groupby(['Date', 'LeadTime'])

  # Create a new column Morethan n members
  IS['MoreMembers'] = False
  for (date, leadtime), group in grouped_data:
      member_count = len(group['Member'].tolist())
      print(member_count,n,member_count>n)
      # If the number of members is greater than n, enter True for the corresponding lines.
      if member_count > n:
          print(n)
          IS.loc[(IS['Date'] == date) & (IS['LeadTime'] == leadtime), 'MoreMembers'] = True
  print(IS[IS['MoreMembers']==False])
  filtered_df = IS[IS['MoreMembers'] == True].copy()
  print(IS[IS['MoreMembers'] == True].copy())
  final_df = pd.DataFrame()
  print(filtered_df)
  grouped = filtered_df.groupby(['Date', 'LeadTime'])
  print(grouped)
  for (date, leadtime), group in grouped:
      print(group)
      current_members = group['Member'].tolist()
      num_current_members = len(current_members)      
      if num_current_members < 16:
          additional_rows = labels[(labels['Date'] == date) & (labels['LeadTime'] == leadtime)]
          additional_rows = additional_rows[~additional_rows['Member'].isin(current_members)]
          num_needed = 16 - num_current_members
          additional_rows = additional_rows.head(num_needed)
          group = pd.concat([group, additional_rows])

      # Ajouter le groupe au DataFrame final
      final_df = pd.concat([final_df, group])

  # Réinitialiser les index du DataFrame final
  final_df.reset_index(drop=True, inplace=True)
  # final_df.to_csv(f"{params.load_directory}/IS_bootstrap_no_duplicate_rr_cumul_correct_16_members.csv")
  final_df.to_csv(f"./IS_bootstrap_no_duplicate_rr_cumul_correct_16_members.csv")
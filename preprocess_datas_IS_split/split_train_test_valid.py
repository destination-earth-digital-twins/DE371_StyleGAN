import pandas as pd


  "This file allows you to split the dataset into train/test/validation. It takes a csv with dates. 
  "It returns --> the training set in the defined interval, the same for the test set and takes one week per month for the validation set.”
  
  def split_csv_with_validation_first(input_file, train_file, test_file, valid_file, train_start_date, train_end_date, test_start_date, test_end_date):
    # Read the CSV file    df = pd.read_csv(input_file)
    
    # Convert the 'DATE' column to datetime format    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Extract one week per month for the validation set
    validation_data = df.groupby(df['Date'].dt.to_period("M")).apply(
        lambda x: x[x['Date'].dt.isocalendar().week == x['Date'].dt.isocalendar().week.iloc[0]]
    ).reset_index(drop=True)
    validation_data['Date'] = validation_data['Date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Create a set of dates used for the validation set to avoid duplicates

    validation_dates = set(validation_data['Date'])
    
        # Filter the remaining data by excluding the dates used for validation

    remaining_data = df[~df['Date'].isin(validation_dates)]

    remaining_data['Date'] = remaining_data['Date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    
    
    # Filter data for train and test based on the date intervals
    train_data = remaining_data[(remaining_data['Date'] >= train_start_date) & (remaining_data['Date'] <= train_end_date)]
    test_data = remaining_data[(remaining_data['Date'] >= test_start_date) & (remaining_data['Date'] <= test_end_date)]
    
    # Save results 
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    validation_data.to_csv(valid_file, index=False)
    
    print(f"Validation data saved to {valid_file} with {len(validation_data)} records.")
    print(f"Train data saved to {train_file} with {len(train_data)} records.")
    print(f"Test data saved to {test_file} with {len(test_data)} records.")

# Example
input_file = "IS_boostrap_no_duplicate_rr_cumul_correct.csv"           
train_file = "IS_boostrap_no_duplicate_rr_cumul_correct_train.csv"     
test_file = "IS_boostrap_no_duplicate_rr_cumul_correct_test.csv"       
valid_file = "IS_boostrap_no_duplicate_rr_cumul_correct_valid.csv"     

train_start_date = "2020-06-15"
train_end_date = "2021-06-01"
test_start_date = "2021-06-01"
test_end_date = "2021-11-12"

split_csv_with_validation_first(input_file, train_file, test_file, valid_file, train_start_date, train_end_date, test_start_date, test_end_date)

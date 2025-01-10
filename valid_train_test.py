import pandas as pd

"""

This file takes a csv that has been made with Importance sampling and bootstrapping. 
In this initial file, for a given date and leadtime, some members are missing. 
So  we return a new csv file with all the missing members if the number of members has exceeded the specified threshold. 

"""

def add_more_than_members_column(df, group_by_cols, threshold, column_name):
    """
    Add column indicating if a member_count has exceeded the threshold.
    """
    df[column_name] = False
    grouped_data = df.groupby(group_by_cols)
    for (date, leadtime), group in grouped_data:
        member_count = len(group['Member'].unique())
        if member_count > threshold:
            df.loc[(df['Date'] == date) & (df['LeadTime'] == leadtime), column_name] = True
    return df

def complete_with_labels(filtered_df, labels, group_by_cols, max_members):
    """
    Complete if members are missing 
    """
    final_df = pd.DataFrame()
    grouped = filtered_df.groupby(group_by_cols)

    for (date, leadtime), group in grouped:
        current_members = group['Member'].tolist()
        num_current_members = len(current_members)

        if num_current_members < max_members:
            additional_rows = labels[(labels['Date'] == date) & (labels['LeadTime'] == leadtime)]
            additional_rows = additional_rows[~additional_rows['Member'].isin(current_members)]
            num_needed = max_members - num_current_members
            additional_rows = additional_rows.head(num_needed)
            group = pd.concat([group, additional_rows])

        final_df = pd.concat([final_df, group])

    final_df.reset_index(drop=True, inplace=True)
    return final_df


# Pipeline
def main():
    # Paths 
    csv_path = 'IS_boostrap_no_duplicate_rr_cumul_correct_.csv'
    labels_path = 'original_csv_path'
    output_path = 'IS_csv_sorted_nsample_threshold.csv'

    #Load datas
    IS_csv= pd.read_csv(csv_path)
    labels = pd.read_csv(labels_path)

    # add the column with the threshold
    column_name = 'MoreThan8Members'
    IS_csv = add_more_than_members_column(IS_csv, ['Date', 'LeadTime'], 8, column_name) # Replace IS_csv by the one you want to modify 

    # Filtered lines where MoreThan8Members is True
    filtered_df = IS_csv[IS_csv['MoreThan8Members'] == True].copy()

    # Complete the missing members with the original csv labels.csv
    final_df = complete_with_labels(filtered_df, labels, ['Date', 'LeadTime'], 16)

    # Save the final dataframe into csv file 
    final_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()

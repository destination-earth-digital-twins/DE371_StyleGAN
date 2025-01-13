import os
import random as rd
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import glob
try:
    from called.utile import make_save_dir, print_progress, print_progress_bar
except ModuleNotFoundError:
    from utile import make_save_dir, print_progress, print_progress_bar

def random_select(proba: float) -> bool:
    """Choice to save sample or not."""
    return random.random() < proba

def create_dirs(save_dir, param):
    """Create directories and csv files for each instance

    Args:
        save_dir (str): the save directory
        param (argparse.Namespace): args of the program
    """
    for instance in range(1, param.n_instances + 1):
        save_dir_instance = f"{save_dir}INST{instance}/"
        make_save_dir(save_dir_instance, param)
        with open(f"{save_dir_instance}{param.output_csv}", "w", encoding="utf8") as file:
            file.write(f"Name,Date,Leadtime,Member,Gigafile,Localindex,Importance\n")

def compute_c(s_rr, q_min, m, l_c):
    print(s_rr,q_min,m,l_c)
    filter_func = lambda c: m + ((q_min - m) / np.tanh(-s_rr / c)) * np.tanh((1 - s_rr) / c) - 1
    l_c = np.abs(fsolve(filter_func, l_c))
    if np.abs(l_c[0] - l_c[1]) < 0.01:
        return l_c[0]
    raise RuntimeError(f"fsolve didn't converge: {l_c}")

def filter(x, s_rr, q_min, m, c):
    return m + ((q_min - m) / np.tanh(-s_rr / c)) * np.tanh((x - s_rr / c))
    
def rough_filter(x, s_rr, q_min, m):
    return m * (x >= s_rr) + q_min * (x < s_rr)

def ravuri_filter(x, s_rr, q_min, m):
    return q_min + m * (1 - np.exp(-x/s_rr))

def importance(grid, parameters, gridshape, variable, param):
    """Compute the importance of a grid

    Args:
        grid (numpy.array): a numpy array grid with several channels of shape = [4, x, y]
        parameters (tuple[float]): Parameters of importance sampling (s_rr, q_min, m, c)
        param (argparse.Namespace): args of the program
    Returns:
        float: the importance.
    """
    if param.verbose >= 5 and not param.progress_bar: print(f"Computing importance...")
    s_rr, q_min, m, c = parameters
    if param.rough:
        i_grid = rough_filter(grid[variable], s_rr, q_min, m)
    elif param.ravuri:
        i_grid = ravuri_filter(grid[variable], s_rr, q_min, m)
    else:
        i_grid = filter(grid[variable], s_rr, q_min, m, c)
    grid_size = gridshape[0]*gridshape[1] # shape = [4, x, y]
    return np.sum(i_grid) / grid_size

def sample_from_instance(save_dir, p_importance, row, param):
    """For each instance, sample and writes data in the csv file

    Args:
        save_dir (str): the save directory
        p_importance (float): the importance computed
        row (_type_): A row from a dataframe
        param (argparse.Namespace): args of the program
    """
    if param.verbose >= 5 and not param.progress_bar: print(f"Sampling...")
    for instance in range(1, param.n_instances + 1):
        save_dir_instance = f"{save_dir}INST{instance}/"
        p_uniform = rd.uniform(0, 1)
        if p_uniform <= p_importance:
            with open(f"{save_dir_instance}{param.output_csv}", "a", encoding="utf8") as file:
                file.write(f"{row['Name']},{row['Date']},{row['LeadTime']},{row['Member']},{row['Gigafile']},{row['Localindex']},{p_importance}\n")

# def importance_sampling(parameters, dirs, gridshape, variable, param):
#     """Compute importance sampling with parameters parameters

#     Args:
#         parameters (tuple[float]): Parameters of importance sampling (s_rr, q_min, m, c)
#         dirs (str): Directories with which data interact (csv_dir, data_dir, save_dir)
#         param (argparse.Namespace): args of the program
#     """
#     if param.verbose >= 1: print(f"Importance sampling...")
#     csv_dir, data_dir, save_dir = dirs
#     create_dirs(save_dir, param)
#     print("HEHOOOOO",f"{csv_dir}labels.csv")

#     dataframe = pd.read_csv(f"{csv_dir}labels.csv")
#     print(f"{csv_dir}labels.csv")
#     s_gigafile = {gigafile  for gigafile in os.scandir(data_dir) if gigafile.name != "labels.csv"}
#     n_gigafile = len(s_gigafile)
#     start_time = perf_counter()
#     for idx_gigafile, gigafile in enumerate(s_gigafile):
#         if param.verbose >= 2:
#             print(f"Loading patch {gigafile.name} ({idx_gigafile + 1}/{n_gigafile})...")
#             if (idx_gigafile + 1) % ((n_gigafile // param.refresh) + 1) == 0:
#                 print_progress(idx_gigafile, n_gigafile, start_time)
#         l_grid = np.load(f"{gigafile.path}")    
#         dataframe_gigafile = dataframe.groupby("Gigafile").get_group(int(gigafile.name[:-4]))
#         for idx_grid, grid in enumerate(l_grid):
#             if param.progress_bar: print_progress_bar(idx_grid, len(l_grid))
#             p_importance = importance(grid, parameters, gridshape, variable, param)
#             sample_from_instance(save_dir, p_importance, dataframe_gigafile.iloc[idx_grid], param)
#         del l_grid
#     if param.verbose >= 2: print(f"All gigafiles processed.")
#     if param.verbose >= 1: print(f"Importance sampling for parameters {parameters} DONE.")

def importance_sampling(parameters, dirs, gridshape, variable, param):
    """Compute importance sampling with parameters parameters

    Args:
        parameters (tuple[float]): Parameters of importance sampling (s_rr, q_min, m, c)
        dirs (str): Directories with which data interact (csv_dir, data_dir, save_dir)
        param (argparse.Namespace): args of the program
    """
    if param.verbose >= 1: print(f"Importance sampling...")
    csv_dir, data_dir, save_dir = dirs
    create_dirs(save_dir, param)
    
    # Load the dataframe
    dataframe = pd.read_csv(f"{csv_dir}labels.csv")
    # s_gigafile = {gigafile for gigafile in os.scandir(data_dir) if gigafile.name != "labels.csv"}
    s_gigafile = {gigafile for gigafile in os.scandir(data_dir) if not gigafile.name.endswith('.csv')}

    n_gigafile = len(s_gigafile)
    start_time = perf_counter()

    # Create a list to store rows that satisfy the condition
    selected_samples = []
    print("LA")
    for idx_gigafile, gigafile in enumerate(s_gigafile):
        print(gigafile.name)
        if param.verbose >= 2:
            print(f"Loading patch {gigafile.name} ({idx_gigafile + 1}/{n_gigafile})...")
            if (idx_gigafile + 1) % ((n_gigafile // param.refresh) + 1) == 0:
                print_progress(idx_gigafile, n_gigafile, start_time)

        # Load the grid data
        l_grid = np.load(f"{gigafile.path}", allow_pickle=True)
        dataframe_gigafile = dataframe.groupby("Gigafile").get_group(int(gigafile.name[:-4]))

        for idx_grid, grid in enumerate(l_grid):
            if param.progress_bar:
                print_progress_bar(idx_grid, len(l_grid))

            # Compute importance sampling probability
            p_importance = importance(grid, parameters, gridshape, variable, param)

            # Perform uniform sampling and save the selected samples
            p_uniform = rd.uniform(0, 1)
            if p_uniform <= p_importance:
                sample = dataframe_gigafile.iloc[idx_grid]
                selected_samples.append(sample)
                if param.verbose >= 3:
                    print(f'SAMPLE {sample} JE GARDE INDEX {idx_grid}')

            # Save the sample from the current instance
            sample_from_instance(save_dir, p_importance, dataframe_gigafile.iloc[idx_grid], param)

        del l_grid

    # Create a new dataframe from selected samples
    selected_dataframe = pd.DataFrame(selected_samples)

    # Save the new dataframe to a CSV file
    selected_dataframe.to_csv(f"./selected_samples.csv", index=False)

    if param.verbose >= 2:
        print(f"All gigafiles processed.")
    if param.verbose >= 1:
        print(f"Importance sampling for parameters {parameters} DONE.")





def bootstrap(IS_csv_folder):
    """ takes n csv files and return one csv without duplicated
    Args:
    IS_csv_folder : dir where csv files with importance sampling are saved
    """
    
    # List to stock dataframes 
    dataframes = []
    # Load csv 
    print('JE SUIS',f"{IS_csv_folder}*.csv")
    for file in glob.glob(f"{IS_csv_folder}/*.csv"):
        print("FILE",file)
        df = pd.read_csv(file)
        dataframes.append(df)

    # Combine all dataframes
    df_combined = pd.concat(dataframes, ignore_index=True)

    # Remove duplicated based on the Name column 
    df_unique = df_combined.drop_duplicates(subset=['Name'])

    #Save the final Dataframe 
    df_unique.to_csv(f'{IS_csv_folder}/IS_boostrap_rr_cumul_correct.csv', index=False)

    print("The final CSV file without duplicates has been created successfully.")    

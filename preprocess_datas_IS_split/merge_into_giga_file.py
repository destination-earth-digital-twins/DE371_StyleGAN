import glob
import os
import numpy as np
from called.utile import make_save_dir, print_progress, print_progress_bar
import pandas as pd
from time import perf_counter
import numpy as np 

def merge_into_gigafiles(data_dir, datatype, args):
    """Merge the numerous little files in gigafiles to load patches at once and accelerate data computation like importance sampling or cropping

    Args:
        working_dir (str): Working directory
        datatype (str): "splitted" or "cropped"
        args (argparse.Namespace): args of the program
    """
    
    s_datatype = {"splitted", "cropped"}
    d_datatype_max_file_loaded = {"splitted": 1000, "cropped": 8000}
    if datatype not in s_datatype:
        raise ValueError(
            "Datatype must be in " + str(s_datatype) + " you gave '" + datatype + "'."
        )
    dataframe = pd.read_csv(f"{data_dir}/")

    index_string = "_".join([str(index) for index in args.crop_indexes])
    if datatype == "cropped":
        data_dir = f"{data_dir}{datatype}_{index_string}"
    else:
        data_dir = f"{data_dir}"
    save_dir = f"{data_dir}_giga/"
    data_dir += f"/"
    make_save_dir(save_dir, args)
    with open(f"{save_dir}labels.csv", "w", encoding="utf8") as file:
        file.write("Name,Date,Leadtime,Member,Gigafile,Localindex\n")
    handle_patch(
        dataframe, data_dir, save_dir, d_datatype_max_file_loaded[datatype], args
    )


def handle_patch(dataframe, data_dir, save_dir, max_files_loaded, args):
    """Handle patch processing.

    Args:
        dataframe (pandas.DataFrame): Dataframe where are saved the name of the files.
        data_dir (str): Data directory
        save_dir (str): Directory where the data is saved
        max_files_loaded (int): Maximum of files loaded in a patch (to prevent overuse of RAM)
        args (argparse.Namespace): args of the program
    """
    n_files = len(dataframe)
    n_patch = n_files // max_files_loaded + 1
    begin = 0
    end = min(max_files_loaded, n_files)
    for patch in range(1, n_patch + 1):
        if args.verbose >= 2:
            print(f"Patch {patch}/{n_patch}")
        giga = load(dataframe, data_dir, save_dir, begin, end, patch, args)
        np.save(f"{save_dir}{patch}.npy", giga)
        del giga
        begin = end
        files_processed = patch * max_files_loaded
        end = min(begin + max_files_loaded, begin + n_files - files_processed)


def load(dataframe, data_dir, save_dir, beg, end, patch, args):
    """Load patch of files from their name in the dataframe and store the data in a list.

    Args:
        dataframe (pandas.DataFrame): the dataframe from which we can find the name of the datafiles
        data_dir (str): Data directory
        save_dir (str): Directory where the data is saved
        beg (int): from which index of the dataframe we want to load the datafiles...
        end (int): ... to where we want to stop
        patch_count (int): the patch number, allowing to save the gigafile number where is saved the numpy array
        args (argparse.Namespace): args of the program

    Returns:
        list: list of numpy arrays representing maps with 4 channels (rr, u, v, t2m)
    """
    n_tot = end - beg
    if args.verbose >= 3:
        print(f"Loading data, {n_tot} files...")
    data = []
    start_time = perf_counter()
    for index in range(beg, end):
        row = dataframe.iloc[index]
        print('ROW',row)
        if args.verbose >= 3 and (index + 1) % (n_tot // args.refresh) == 0:
            print_progress(index - beg, n_tot, start_time)
        data.append(np.load(f"{data_dir}{row['Name']}.npy"))

        with open(f"{save_dir}labels.csv", "a", encoding="utf8") as file:
            file.write(
                f"{row['Name']},{row['Date']},{row['LeadTime']},{row['Member']},{patch},{index-beg}\n"
            )
    return data
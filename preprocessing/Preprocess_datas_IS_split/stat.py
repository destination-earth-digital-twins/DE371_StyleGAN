import concurrent.futures
import json
import os
import shutil
from argparse import ArgumentParser
from collections import Counter
from multiprocessing.pool import ThreadPool as Pool
from time import perf_counter
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from utils import (make_save_dir, parse_float, print_progress,
                          print_progress_bar)
from matplotlib import pyplot as plt


def compute_area_greater_than(variable, data_dir, gridshape,threshold_list, args):
    """Extract from each grid all the values greater than threshold and store them in a list

    Args:
        variable (int): index of the channels corresponding to the variable
        data (list): list of the loaded data
        args (argparse.Namespace): args of the program

    Returns:
        list[float]: store every value greater than the threshold
    """
    gigafile_set = {gigafile for gigafile in os.scandir(data_dir) if gigafile.name not in ["INST1","area_proportions.npy", "area_proportions.png", "labels.csv"]}
    n_gigafiles = len(gigafile_set)
    x_length, y_length = gridshape[0], gridshape[1]
    l_mean_proportion = np.zeros([len(threshold_list)])
    start_time = perf_counter()
    n_grid = 0
    for idx_gigafile, gigafile in enumerate(gigafile_set):
        if (idx_gigafile + 1) % ((n_gigafiles // args.refresh) + 1) == 0:
            print_progress(idx_gigafile, n_gigafiles, start_time)
        print(f"Loading gigafile {gigafile.name} ({idx_gigafile + 1}/{n_gigafiles})")
        l_grid = np.load(gigafile.path)
        n_grid += len(l_grid)
        for idx_threshold, threshold in enumerate(threshold_list):
            if args.verbose >= 4: print_progress_bar(idx_threshold, len(threshold_list))
            for idx, grid in enumerate(l_grid):
                mask = grid[variable] > threshold
                extracted_values = grid[variable][mask]
                l_mean_proportion[idx_threshold] += len(extracted_values)
    l_mean_proportion /= (gridshape[0] * gridshape[1] * n_grid)
    return l_mean_proportion



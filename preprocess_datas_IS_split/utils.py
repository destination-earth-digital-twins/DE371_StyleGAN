import os
import random as rd
from time import perf_counter

import numpy as np


def print_progress(index, n_tot, start_time):
    """Print the progress and the estimated remaining time

    Args:
        index (int): the iteration in which the function is called
        n_tot (int): the total number of itertion
        start_time (float): time at which the loop was started
    """
    progress = 100 * index / n_tot
    elapsed_time = perf_counter() - start_time
    if progress != 0:
        print(f"\033[92m\n{progress}% in {elapsed_time}s...")
        print(
            f"Remaining around {(100 - progress) * elapsed_time / progress}s\033[0m\n"
        )


def parse_float(str_float):
    str_float = [char.replace(".", "-") for char in str_float]
    return "".join(str_float)


def make_save_dir(save_dir, args):
    if not os.path.exists(save_dir) and not os.path.exists(save_dir + "_done"):
        if args.verbose >= 6:
            print(f"creating save dir {save_dir}")
        os.makedirs(save_dir)

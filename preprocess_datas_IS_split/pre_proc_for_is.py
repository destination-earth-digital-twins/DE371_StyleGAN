from argparse import ArgumentParser
from merge_into_giga_file import load, merge_into_gigafiles,handle_patch
import pandas as pd
from time import perf_counter
import numpy as np 


if __name__ == "__main__":
    ## ARGPARSE ##
    parser = ArgumentParser()

    parser.add_argument("--save_directory",type=str,default="/project/home/p200177/DE_371/datasets/dataset_Meteo_France_rr_u_v_t2m/data/IS_rr_debug_1_1.0_0_0_0_0_0_256_large_lt",help="Directory where data will be saved: 'pre_proc_' + save_directory",
    )
    parser.add_argument("-l","--load_directory",type=str,default="/project/home/p200177/DE_371/datasets/dataset_Meteo_France_rr_u_v_t2m/data/IS_rr_debug_1_1.0_0_0_0_0_0_256_large_lt",help="Data directory from which data is loaded",
    )
    parser.add_argument("-c", "--crop", action="store_true", help="Crop while processing all"
    )
    parser.add_argument("--crop_indexes",type=int,nargs="*",default=[120, 376, 540, 796],help="Crop index. If not specified, take the values : [120, 376, 540, 796] (SE_indexes). If no crop is wanted, pass 0 as an argument. Ex: --crop_indexes 120 376 540 796 for SE_indexes; --crop_indexes 0 for no crop",
    )

    parser.add_argument("-r","--refresh",type=int,default=5,help="Frequence at which progress is shown",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity"
    )

    parser.add_argument("--split_dir", type=str, help="When using split() on priam")

    params= parser.parse_args()

    ## GLOBAL ##
    variable_name_list = ["rr", "u", "v", "t2m"]
    INDEXES = (
        params.crop_indexes
    )  # SE_INDEXES = (120, 376, 540, 796); FR_INDEXES = (20, 680, 150, 972)
    CROPPING = len(INDEXES) == 4
    RAW_DATA_DIR = params.load_directory
    SAVE_DIR = f"{params.save_directory}/"
    
    #Gigafile and labels.csv
    merge_into_gigafiles(SAVE_DIR, "splitted", params)
    
    


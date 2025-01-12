from argparse import ArgumentParser
import pandas as pd
from time import perf_counter
import numpy as np 
from merge_into_giga_file import load, merge_into_gigafiles,handle_patch

from process_is import importance_sampling, compute_c, bootstrap


if __name__ == "__main__":
    parser = ArgumentParser()

# parser.add_argument("--directory", type=str, default='/data_for_importance_sampling/',help="Data directory from which data is loaded")
# parser.add_argument("--main_path", type=str, default="/project/home/p200177/DE_371/datasets/dataset_Meteo_France_rr_u_v_t2m/data", help="Base path")
# parser.add_argument("--save", type=str, default='saveds',help="Data directory where the data is saved")
# parser.add_argument("--l_c", type=float, nargs="*", help="The initial points for fsolve. MUST BE CLOSE TO THE ROOT")
# parser.add_argument("--default_param", type=float, default = [5, 0.001, 500], help=f"Importance sampling parameters.")
# parser.add_argument("-b", "--progress_bar", action="store_true", help="Print the progress bar")
# parser.add_argument("-o", "--rough", action="store_true", help="If used, importance sampling is done with a rough ladder filter.")
# parser.add_argument("--n_instances", type=int, default=1, help="Number of instances")
# parser.add_argument("--ignore_c", action="store_true", help="Don't execute fsolve to find c")
# parser.add_argument("--ravuri", action="store_true", help="Importance sampling with the same function as Ravuri et al.")
# parser.add_argument("--output_csv", type=str, default= 'IS_labels.csv',help= 'new csv file with IS')
# ######################################
# parser.add_argument("--force", action="store_true")
# parser.add_argument("--old_param", type=str)
# parser.add_argument("--bootstrap", type=int, default=1, help="boostrap or not")

    parser.add_argument("--save_directory",type=str,default="./test1/",help="Directory where data will be saved: 'pre_proc_' + save_directory",
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


    params= parser.parse_args()


    SAVE_DIR = f"{params.save_directory}/"
    
    #Gigafile and labels.csv
    merge_into_gigafiles(SAVE_DIR, "splitted", params)
    
    

# from argparse import ArgumentParser

# from process_is import importance_sampling, compute_c, bootstrap

# # nohup python3 -u main.py -vv -r 5 -p 5 0.001 500 --n_instances 50 pre_proc_31-07-10h/cropped_giga/ 11-08-11h_default/ > output/default.txt 2> output/default.err &

# #### ARGPARSE ####
# parser = ArgumentParser()

# ######################################

# param = parser.parse_args()

# #### GLOBAL ####
# ## IMPORTANCE SAMPLING ##
# if param.l_c is None:
#     param.l_c = [1, 1.25]
# S_RR, Q_MIN, M = param.default_param
# print("S_RR, Q_MIN, M ",S_RR, Q_MIN, M )
# C = compute_c(S_RR, Q_MIN, M, param.l_c)
# print(f"c = {C}")

# PARAMETERS = (S_RR, Q_MIN, M, C)
# PARAMETERS_STR = f"{S_RR}_{Q_MIN}_{M}"

# ## STATS ##
# VARIABLE= f"rr"
# THRESHOLD = param.threshold
# GRIDSHAPE = (256, 256)

# VAR_NAMES = (f"rr", f"u", f"v", f"t2m")
# VARIABLE= VAR_NAMES.index(VARIABLE)
# THRESHOLD_STR = f"{THRESHOLD}"
# #### PATH ####
# print('je suis ici')
# MAIN_PATH = f"{param.main_path}"
# DIRECTORY = f"{param.directory}"

# #### IMPORTANCE SAMPLING ####
# ## PATH ##
# CSV_DIR = f"{MAIN_PATH}{DIRECTORY}"
# DATA_DIR = f"{MAIN_PATH}{DIRECTORY}"
# SAVE_DIR = f"{MAIN_PATH}{param.save}{PARAMETERS_STR}___4/"
# save_dir = './saved'
# DIRS = (CSV_DIR, DATA_DIR, save_dir)

# if param.bootstrap==1:
#     importance_sampling(PARAMETERS, DIRS, GRIDSHAPE, VARIABLE, param)
# else:
#     for number_csv in range(param.bootstrap):
#         param.output_csv=f"IS_labels_{number_csv}.csv"
#         importance_sampling(PARAMETERS, DIRS, GRIDSHAPE, VARIABLE, param)

# print(f"DONE")

# bootstrap('/home/users/u101957/DE371_StyleGAN/preprocess_datas_IS_split/savedINST1/')

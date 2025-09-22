from argparse import ArgumentParser
import pandas as pd
from time import perf_counter
import numpy as np 
from merge_into_giga_file import load, merge_into_gigafiles,handle_patch
from split_train_test_valid import  More_than, split_csv_with_validation_first
from importance_sampling_bootstrap import importance_sampling, compute_c, bootstrap
from stat import compute_area_greater_than

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-r", "--refresh", type=int, default=25, help="Progress is shown 'refresh' times")
    parser.add_argument("-t", "--threshold", type=float, default=0, help="Threshold for stats")


    parser.add_argument("--giga_directory", type=str, help="Data directory where gigafile are saved")
    parser.add_argument("--main_path", type=str, help="Base path")
    parser.add_argument("--data_directory",type=str,help="Data directory from which data is loaded")
    
    parser.add_argument("-c", "--crop", action="store_true", help="Crop while processing all")
    parser.add_argument("--crop_indexes",type=int,nargs="*",default=[120, 376, 540, 796],help="Crop index. If not specified, take the values : [120, 376, 540, 796] (SE_indexes). If no crop is wanted, pass 0 as an argument. Ex: --crop_indexes 120 376 540 796 for SE_indexes; --crop_indexes 0 for no crop",)
    parser.add_argument("--l_c", type=float, nargs="*", help="The initial points for fsolve. MUST BE CLOSE TO THE ROOT")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="verbose parameter to control the level of detail in program output (higher is more detailed).")
    parser.add_argument("--default_param", type=float, default = [5, 0.001, 500], help=f"Importance sampling parameters.")
    parser.add_argument("-b", "--progress_bar", action="store_true", help="Print the progress bar")
    parser.add_argument("-o", "--rough", action="store_true", help="If used, importance sampling is done with a rough ladder filter.")
    parser.add_argument("--n_instances", type=int, default=1, help="Number of instances")
    parser.add_argument("--ignore_c", action="store_true", help="Don't execute fsolve to find c")
    parser.add_argument("--ravuri", action="store_true", help="Importance sampling with the same function as Ravuri et al.")
    parser.add_argument("--output_csv", default='new.csv',type=str,help= 'new csv file with IS and bootstrap')
    parser.add_argument("--origin_csv", type=str, help= 'original csv before importance sampling')

    ######################################
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--old_param", type=str)
    parser.add_argument("--n_bootstrap", type=int, default=4, help="number boostrap")
    parser.add_argument("--bootstrap", action = "store_true", help="apply boostrap or not")
    parser.add_argument("--n_ensemble", type=int, default=4, help="number boostrap")
    parser.add_argument('--method_type', type=str, choices=["stat","importance_sampling","merge_into_giga_file", "complete_members","split_dataset"], help='Type of methods we want to apply ')



    params= parser.parse_args()
    
    ##MERGE INTO GIGAFILE:
    if params.method_type=="merge_into_giga_file":
        merge_into_gigafiles( "splitted", params)
        
    
##    IMPORTANCE SAMPLING 
    if params.method_type=='importance_sampling':
        if params.l_c is None:
            params.l_c = [1, 1.25]
        S_RR, Q_MIN, M = params.default_param
        C = compute_c(S_RR, Q_MIN, M, params.l_c)
        
        VARIABLE= f"rr"
        VAR_NAMES = (f"rr", f"u", f"v", f"t2m")
        VARIABLE= VAR_NAMES.index(VARIABLE)
        PARAMETERS = (S_RR, Q_MIN, M, C)
        PARAMETERS_STR = f"{S_RR}_{Q_MIN}_{M}"
        GRIDSHAPE = (256, 256)

        #### IMPORTANCE SAMPLING ####
        CSV_DIR = f"{params.main_path}{params.giga_directory}"
        DATA_giga_DIR = f"{params.main_path}{params.giga_directory}"
        SAVE_DIR= f"{params.main_path}{params.giga_directory}"
        DIRS = (CSV_DIR, DATA_giga_DIR, SAVE_DIR)

        if not params.bootstrap:
            importance_sampling(PARAMETERS, DIRS, GRIDSHAPE, VARIABLE, params)
        else:
            for number_csv in range(params.n_bootstrap):
                params.output_csv=f"IS_labels_{number_csv}.csv"
                importance_sampling(PARAMETERS, DIRS, GRIDSHAPE, VARIABLE, params)
          bootstrap(CSV_DIR,params)
        print(f"DONE")



#     # ADD missing dates
    if params.method_type=="complete_members":
        More_than(8,params)

    #SPLIT 
    if params.method_type=="split_dataset":
        
        train_start_date="2020-06-15"
        train_end_date="2021-06-01"   
        test_start_date = "2021-06-01"
        test_end_date = "2021-11-12"

        split_csv_with_validation_first(params,train_start_date,test_end_date,test_start_date)

    #COMPUTE AND PLOT AREA PROPORTION FOR GIGAFILES:
    
    if params.method_type=="stat"
        l_thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30]
        l_mean = compute_area_greater_than(0, DATA_giga_DIR, [256, 256],  l_thresholds,params)
        np.save(f"{DATA_giga_DIR}area_proportions.npy", l_mean)
        l_mean = np.load(f"{DATA_DIR}area_proportions.npy")
        plt.plot(l_thresholds, l_mean)
        plt.yscale("log")
        plt.xlabel("s_rr")
        plt.ylabel("Area proportion (log10)")
        plt.title("Area proportion for precipitation >= s_rr")
        plt.savefig(f"{DATA_DIR}area_proportionslog10.png")
        plt.clf()
        plt.plot(l_thresholds, l_mean)
        plt.xlabel("s_rr")
        plt.ylabel("Area proportion (log10)")
        plt.title("Area proportion for precipitation >= s_rr")
        plt.savefig(f"{DATA_DIR}area_proportions.png")


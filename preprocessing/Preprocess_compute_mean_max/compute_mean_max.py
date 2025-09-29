import numpy  as np
from glob import glob
from time import perf_counter
import os

var_dict={'rr' : 0, 'u' : 1, 'v' : 2, 't2m' :3 , 'orog' : 4, 'z500': 5, 't850': 6, 'tpw850': 7}

def compute_mean_max(data_path, fname="_with_8_var"):
    mean_fname = f"mean{fname}.npy"
    max_fname = f"max{fname}.npy"
        
    filenames = glob(data_path + "_samp*")
    assert len(filenames) > 60000, "there should be more than 60k arome samples"

    sum_arr = 0
    max_arr = -1e15
    min_arr = 1e15 # to be sure nothing is bugged
    t_s = perf_counter()
    for i, file in enumerate(filenames):
        if (i+1)%10000==0:
            print(f"{i+1}/{len(filenames)}: {perf_counter()-t_s}")
            t_s = perf_counter()
        arr = np.load(file)
        tmp_max = arr.max(axis=(-1,-2))
        tmp_min = arr.min(axis=(-1,-2))
        #assert tmp_max.shape[0]==8
        max_arr = np.maximum(max_arr, tmp_max)
        min_arr = np.minimum(min_arr, tmp_min)
        #assert max_arr.shape[0]==8
        sum_arr += arr
    mean_arr = sum_arr.mean(axis=(-1,-2))
    print("min of arome: ", min_arr)
    print("max of arome: ", max_arr)
    #assert mean_arr.shape[0]==8
    mean_arr /= len(filenames)
    np.save(data_path + mean_fname, mean_arr)
    Max = np.maximum(np.abs(max_arr-mean_arr), np.abs(min_arr-mean_arr))
    np.save(data_path + max_fname, Max)



def new_database_no_mean(data_orig_path, new_data_path, vars=['z500']):
    orig_files = np.sort(glob(data_orig_path+"_samp*"))
    assert len(orig_files) > 60000, "there should be more than 60k arome samples"

    t_s = perf_counter()
    for i, file in enumerate(orig_files):
        if (i+1)%10000==0:
            print(f"{i+1}/{len(orig_files)}: {perf_counter()-t_s}")
            t_s = perf_counter()
        
        arr = np.load(file)[[var_dict[var] for var in vars]]
        m = np.mean(arr, axis=(-2,-1), keepdims=True)
        new_arr = arr-m
        if i==3 or i==30000: # sanity check
            print("new means: ", np.mean(new_arr, axis=(-2,-1)))
        np.save(new_data_path+f"_sample{file[len(data_orig_path)+7:]}", new_arr)


def new_database_mean_pert(data_orig_path, new_data_path, vars=['z500']):
    orig_files = np.sort(glob(data_orig_path+"_samp*"))
    assert len(orig_files) > 60000, "there should be more than 60k arome samples"

    t_s = perf_counter()
    for i, file in enumerate(orig_files):
        if (i+1)%10000==0:
            print(f"{i+1}/{len(orig_files)}: {perf_counter()-t_s}")
            t_s = perf_counter()
        
        arr = np.load(file)[[var_dict[var] for var in vars]]
        m = np.mean(arr, axis=(-2,-1), keepdims=True)
        new_arr = np.concatenate((arr-m, np.zeros((8,256,256))+m), axis=0)
        if i==3 or i==30000: # sanity check
            print("new means: ", np.mean(new_arr[:len(var_dict)], axis=(-2,-1)))
        np.save(new_data_path+f"_sample{file[len(data_orig_path)+7:]}", new_arr)


if __name__=='__main__':
    mode='mean_pert' # can be either 8_vars, mean_0 or mean_pert

    data_path = ""
    
    if mode=='mean_0':
        new_data_path = ""
    
    if mode=="mean_pert":
        new_data_path = ""
    
    if not os.path.isdir(new_data_path):
        os.mkdir(new_data_path)

    if mode=="mean_0":
        new_database_no_mean(data_orig_path=data_path, new_data_path=new_data_path, vars=list(var_dict.keys()))
        print("\ncomputing the new means/max...\n")
        compute_mean_max(new_data_path, fname="_with_0_mean")
    
    if mode=="mean_pert":
        new_database_mean_pert(data_orig_path=data_path, new_data_path=new_data_path, vars=list(var_dict.keys()))
        print("\ncomputing the new means/max...\n")
        compute_mean_max(new_data_path, fname="_mean_pert")
    
    if mode=="8_vars":
        compute_mean_max(data_path=data_path)
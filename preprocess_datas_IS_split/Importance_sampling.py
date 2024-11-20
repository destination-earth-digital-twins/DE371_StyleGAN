import numpy as np
from numpy.typing import NDArray
import os 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import shutil
import random as rd

cmapRR = colors.ListedColormap(["white","mediumpurple","blue","dodgerblue","darkseagreen","seagreen","greenyellow","yellow", "navajowhite","sandybrown","darkorange","red","darkred","black"], name='from_list', N=None)


def importance(fields: NDArray[np.float32],pmin: float,m: float,s: int) ->  NDArray[np.float32]:
    """ Calculates the probability of saving each sample in a dataset using
    the importance sampling method (see section 2b)

    Args:
        fields (NDArray[np.float32]): rainfall dataset with shape (samples,x,y) 
        pmin (float): minimum probability of saving a sample
        m (float): multiplying factor
        s (int): rainfall interest threshold

    Returns:
        NDArray[np.float32]: probability of saving each sample 
    """
    q=pmin
    c=1.19
    Importance_list=np.empty([fields.shape[0]])
    Importance_list_bis =np.empty([1])

    # for index,k in  enumerate(fields) :
    for index in range(1):
        k = fields
      #  print('index',index)
     #   print('IMPORTNCE', Importance, m+np.mean((m-q)/(np.tanh(s/c))*np.tanh((k-s)/c)))
        Mean_exp=np.mean(1-np.exp(-k/s)) 
        Importance=pmin+m*Mean_exp
        Importance=np.min((1,Importance))
        Importance_bis = np.min((1,m+np.mean((m-q)/(np.tanh(s/c))*np.tanh((k-s)/c))))
        #Importance_bis = m+np.mean((m-q)/(np.tanh(s/c))*np.tanh((k-s)/c))
        Importance_list[index]=Importance

        Importance_list_bis[index]=Importance_bis

    #print(Importance_list)
    return  Importance_list,Importance_list_bis

def random_select(proba: float) -> bool :
    """ Choice to save sample or not

    Args:
        proba (float): probability of saving the sample

    Returns:
        bool: saved or not
    """

    if random.random()<proba :
        print(random.random,True)
        return True
    else :
        print(False)
        return False

random_select_vect=np.vectorize(random_select)

def extract_data(data_RR: NDArray[np.float32],pmin: float,m: float,s: int) -> NDArray[np.float32]:
    """Extract data with importance sampling method

    Args:
        data_RR (NDArray[np.float32]): rainfall dataset
        pmin (float): minimum probability of saving a sample_
        m (float): multiplying factor
        s (int): rainfall interest threshold

    Returns:
        NDArray[np.float32]: dataset with selected samples after importance sampling
    """
    # print(data_RR.shape)

    # data_RR = np.expand_dims(data_RR, axis=0)
    # print(data_RR.shape)
    Imp_vect=importance(data_RR,pmin,m,s)[0]
    print(Imp_vect)

    Imp_vect_bis=importance(data_RR,pmin,m,s)[1]
    Bool = random_select_vect(Imp_vect)
    Bool_bis=random_select_vect(Imp_vect_bis)
    data_RR_select_bis=data_RR[Bool_bis,:,:]
    data_RR_select= data_RR[Bool,:,:]

    return data_RR_select,data_RR_select_bis

from argparse import ArgumentParser

#from stats import run_stat

# nohup python3 -u main.py -vv -r 5 -p 5 0.001 500 --n_instances 50 pre_proc_31-07-10h/cropped_giga/ 11-08-11h_default/ > output/default.txt 2> output/default.err &

DEFAULT_PARAM = [5, 0.001, 500]
#### ARGPARSE ####
parser = ArgumentParser()

parser.add_argument("--directory", type=str, default='/IS_rr_debug_1_1.0_0_0_0_0_0_256_large_lt/',help="Data directory from which data is loaded")
parser.add_argument("--save", type=str, default='saved',help="Data directory where the data is saved")
parser.add_argument("--l_c", type=float, nargs="*", help="The initial points for fsolve. MUST BE CLOSE TO THE ROOT")
parser.add_argument("-r", "--refresh", type=int, default=10, help="Frequence at which progress is shown")
parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
parser.add_argument("-p", "--param", type=float, nargs="*", help=f"Importance sampling parameters. If not specified, take the values : {DEFAULT_PARAM}")
parser.add_argument("-t", "--threshold", type=float, default=0, help="Threshold for stats")
parser.add_argument("-b", "--progress_bar", action="store_true", help="Print the progress bar")
parser.add_argument("-o", "--rough", action="store_true", help="If used, importance sampling is done with a rough ladder filter.")

parser.add_argument("--stats_only", action="store_true", help="If used, only run_stat and save_mix are executed")
parser.add_argument("--main_path", type=str, default="/project/home/p200177/DE_371/datasets/dataset_Meteo_France_rr_u_v_t2m/data", help="Base path")
parser.add_argument("--n_instances", type=int, default=1, help="Number of instances")
parser.add_argument("--ignore_c", action="store_true", help="Don't execute fsolve to find c")
parser.add_argument("--ravuri", action="store_true", help="Importance sampling with the same function as Ravuri et al.")

######################################
parser.add_argument("--force", action="store_true")
parser.add_argument("--old_param", type=str)
######################################

args = parser.parse_args()
if args.param is None:
    args.param = DEFAULT_PARAM
    print(f"Default parameter list: {args.param}")
    
def filter(x, s_rr, q_min, m, c):
    return m + ((q_min - m) / np.tanh(-s_rr / c)) * np.tanh((x - s_rr / c))

def importance(grid, parameters, gridshape, variable):#, args):
    """Compute the importance of a grid

    Args:
        grid (numpy.array): a numpy array grid with several channels of shape = [4, x, y]
        parameters (tuple[float]): Parameters of importance sampling (s_rr, q_min, m, c)
        args (argparse.Namespace): args of the program
    Returns:
        float: the importance.
    """
    # if args.verbose >= 5 and not args.progress_bar: print(f"Computing importance...")
    # s_rr, q_min, m, c = parameters
    # if args.rough:
    #     i_grid = rough_filter(grid[variable], s_rr, q_min, m)
    # elif args.ravuri:
    #     i_grid = ravuri_filter(grid[variable], s_rr, q_min, m)
    s_rr, q_min, m, c = parameters
    i_grid = filter(grid[variable], s_rr, q_min, m, c)
    grid_size = gridshape[0]*gridshape[1] # shape = [4, x, y]
    return np.sum(i_grid) / grid_size

print('ok')
VARIABLE= f"rr"
THRESHOLD = args.threshold
GRIDSHAPE = (256, 256)

VAR_NAMES = (f"rr", f"u", f"v", f"t2m")
VARIABLE= VAR_NAMES.index(VARIABLE)
# data2plot_origin = np.load('/project/scratch/p200177/DE_371/angeliquebonamy/data_basile_inv/samples_AROME_for_AE_1/scenarios/Alpes-Mar_Golfe-G/Rsemble_Alpes-Mar_Golfe-G_1.npy').astype(np.float32)[:,np.newaxis,:,:]#samples_precip/EP_weights_tests/AMSE/inversion/invertFsemble_Rsemble_Rien signif_2.npy_.npy').astype(np.float32)[:,np.newaxis,:,:]
# data2plot_origin = np.load('/project/scratch/p200177/DE_371/angeliquebonamy/data_basile_inv/samples_AROME_for_AE_1/scenarios/Rien_signif/Rsemble_Rien signif_1.npy')
# data2plot_origin = np.load('/project/home/p200177/DE_371/datasets/dataset_Meteo_France_rr_u_v_t2m/data/IS_rr_debug_1_1.0_0_0_0_0_0_256_large_lt/_sample13.npy').astype(np.float32)[:,np.newaxis,:,:]
dir_folder = '/project/home/p200177/DE_371/datasets/dataset_Meteo_France_rr_u_v_t2m/data/IS_rr_debug_1_1.0_0_0_0_0_0_256_large_lt/'
liste = os.listdir(dir_folder)[0:15]
for index,sample in enumerate(sorted(liste)):
 #   print(sample)
    path_sample=os.path.join(dir_folder,sample)
    data2plot_origin = np.load(path_sample)
    print(data2plot_origin.shape)
    args = parser.parse_args()
    p_importance = importance(data2plot_origin,(5, 10**-4, 500,1.19),(256,256),VARIABLE)
    print(p_importance)#,args))
    p_uniform = rd.uniform(0, 1)
    if p_uniform <= p_importance:
        print('SAMPLE',sample,'JE GARDE',index)
    plt.imsave(f'./{index}.png', data2plot_origin[0][0],cmap=cmapRR,origin='lower')
    print(data2plot_origin.shape)
    result = extract_data(data2plot_origin[0],10**-4,500,5)[0]
    result_bis = extract_data(data2plot_origin[0],10**-4,500,5)[1]
    print(index,'INDEX')
    print(result,result_bis)
    # print(result.shape,type(result),result)
    if result.shape[0]>0:
        print('Ca reste')
    else:
        print('ca part')



# import numpy as np
# from numpy.typing import NDArray
# import random
# import os
# import shutil
# from multiprocessing import Pool

# def importance(fields: NDArray[np.float32], pmin: float, m: float, s: int) -> NDArray[np.float32]:
#     """Calculates the probability of saving each sample in a dataset using the importance sampling method."""
#     q = pmin
#     c = 1.19
#     Importance_list = np.empty([fields.shape[0]])
#     for index in range(fields.shape[0]):
#         k = fields[index]
#         Importance =np.min((1,m+np.mean((m-q)/(np.tanh(s/c))*np.tanh((k-s)/c))))
#         Importance_list[index] = min(1, Importance)
#     return Importance_list

# def random_select(proba: float) -> bool:
#     """Choice to save sample or not."""
#     return random.random() < proba

# random_select_vect = np.vectorize(random_select)

# def process_file(args):
#     """Process a single file: extract samples and save selected ones."""
#     src_folder, sample, pmin, m, s, dst_folder = args

#     # Load data using memory mapping
#     path_sample = os.path.join(src_folder, sample)
#     data = np.load(path_sample, mmap_mode='r')  # Use mmap_mode to handle large data efficiently

#     # Apply importance sampling
#     importance_vector = importance(data[0], pmin, m, s)
#     selection_mask = random_select_vect(importance_vector)

#     # Save selected samples based on selection mask
#     if np.any(selection_mask):  # Only process if some samples are selected
#         selected_data = data[0][selection_mask, :, :]
#         save_path = os.path.join(dst_folder, f"selected_{sample}")
#         np.save(save_path, selected_data)  # Save the selected samples
#         print(f"Processed and saved: {save_path}")
#     else:
#         print(f"No samples selected in file: {sample}")

# def main(src_folder: str, dst_folder: str, pmin: float, m: float, s: int, num_workers: int = 4):
#     """Main function to process all files in the source folder with multiprocessing."""
#     if not os.path.exists(dst_folder):
#         os.makedirs(dst_folder)

#     # List all files in the source folder
#     file_list = sorted(os.listdir(src_folder))

#     # Prepare arguments for parallel processing
#     args = [(src_folder, sample, pmin, m, s, dst_folder) for sample in file_list]

#     # Use multiprocessing to process files in parallel
#     with Pool(processes=num_workers) as pool:
#         pool.map(process_file, args)

# if __name__ == "__main__":
#     # Folder paths
#     src_folder = '/path/to/your/source_folder'
#     dst_folder = '/path/to/your/destination_folder'

#     # Parameters for importance sampling
#     pmin = 0.01
#     m = 500
#     s = 100

#     # Number of parallel processes
#     num_workers = 4  # Adjust according to your system's resources

#     # Run the main function
#     main(src_folder, dst_folder, pmin, m, s, num_workers)

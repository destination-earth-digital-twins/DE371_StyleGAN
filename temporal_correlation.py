import gan.data.dataset_handler_ddp as DSH
from torch.utils.data import DataLoader, Dataset
import numpy as np
from expe_init import get_expe_parameters
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
import torch 
import scipy

config = get_expe_parameters().parse_args()
config.stat_folder = ''
config.crop_indexes = [0,256,0,256]
config.crop_size = (256,256)
config.multi_timestep_mode = True
config.nb_timesteps = 44
config.timestep_period = 1
config.stack_sample_along_time_and_variable = False
config.var_names=['rr', 'u', 'v', 't2m']
Dl_train = DSH.ISData_Loader("Train", config)
dataset = DSH.ISDataset(config, Dl_train.dataset_handler_yaml, 'coords', variable_indices=[0,1,2,3], transform=Dl_train.transform())
train_dataloader = DataLoader(dataset = dataset,
                    batch_size = 16,
                    shuffle = False,
                    drop_last = True,
                    num_workers=1)

loop = enumerate(train_dataloader)

# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6,6))


for i, batch in loop:
    img, _, _ = batch
    print(img.size())

    step = 0
    img = img.numpy()
    for var_id in range(4):
        fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(24,24))

        for member_id in range(len(img)):
            pearsons_first_to_each_leadtime = list()
            pearsons_sliding = list()
            for t in range(len(img[0])): 
                pearsons_first_to_each_leadtime.append(scipy.stats.pearsonr(img[member_id][0][var_id].flatten(), img[member_id][t][var_id].flatten()).statistic)
                if t==0 :
                    pearsons_sliding.append(np.nan)
                elif t <= len(img[0])-2:
                    pearsons_sliding.append(scipy.stats.pearsonr(img[member_id][t][var_id].flatten(), img[member_id][t+1][var_id].flatten()).statistic)
            ax[member_id//4][member_id%4].plot(range(len(img[0])), pearsons_first_to_each_leadtime, linewidth=6)
            ax[member_id//4][member_id%4].plot(range(len(img[0])-1), pearsons_sliding, linewidth=6)
            ax[member_id//4][member_id%4].grid(True)
        fig.suptitle(f'Pearson Temporal Correlation on {config.var_names[var_id]} for each member', fontsize=60)
        fig.savefig(f'Pearson_temporal_cor_{config.var_names[var_id]}.png')
        print(f'Pearson_temporal_cor_{config.var_names[var_id]}')




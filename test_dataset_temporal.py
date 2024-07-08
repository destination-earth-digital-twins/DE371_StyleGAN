import gan.data.dataset_handler_ddp as DSH
from torch.utils.data import DataLoader, Dataset
import numpy as np
from main_gan import get_expe_parameters
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
import torch 

def create_frame(fig): 
    r"""Create frames for frame mode

    Args :
           fig : fig of matplotlib

    Return : frame : Image date type

    """
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.int8)
    w, h = fig.canvas.get_width_height()
    data = data.reshape((h, w, 3))
    frame = Image.fromarray(data, "RGB")
    return frame

config = get_expe_parameters().parse_args()
config.crop_indexes = [0,256,0,256]
config.crop_size = (256,256)
config.multi_timestep_mode = True
config.timestep_period=1
print('nb_timesteps',config.nb_timesteps)
config.nb_timesteps = 45
print('nb_timesteps',config.nb_timesteps)
config.var_names=['t2m']

Dl_train = DSH.ISData_Loader("Train", config)
dataset = DSH.ISDataset(config, Dl_train.dataset_handler_yaml, 'coords', variable_indices=[3], transform=Dl_train.transform())
train_dataloader = DataLoader(dataset = dataset,
                    batch_size = 16,
                    shuffle = False,
                    drop_last = True,
                    num_workers=1)

loop = enumerate(train_dataloader)

fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(24,24))


for i, batch in loop:
    img, _, _ = batch
    print(img.size())
    if i > 0 :
        plt.close() 
        raise NotImplementedError
    
    else :
        step = 0
        frames = list()
        for t in trange(len(img[0])):
             # iterate over first member only
            for member_id in range(16):
                ax[member_id%4][member_id//4].imshow(img[member_id][t], origin="lower", clim=(torch.min(img[member_id][t]), torch.max(img[member_id][t])), cmap="coolwarm")
                # ax[member_id//4][member_id%4].set_title("[t2m in K] time : +{}h".format(config.timestep_period*t))
            fig.suptitle(f"leadtime {t*config.timestep_period}")
            frames.append(create_frame(fig))
            step+=1
        frame_one = frames[0]
        frame_one.save(
            '/project/scratch/p200177/DE_371/victorsanchez/results/temporal_gif/' + f"plot_time_step_period_{config.timestep_period}_member_{member_id}.gif",
            format="GIF",
            append_images=frames,
            save_all=True,
            duration=20*step,
            loop=0,
        )





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
config.nb_timesteps=45

Dl_train = DSH.ISData_Loader("Train", config)
dataset = DSH.ISDataset(config, Dl_train.dataset_handler_yaml, 'coords', variable_indices=[3], transform=Dl_train.transform())
train_dataloader = DataLoader(dataset = dataset,
                    batch_size = 16,
                    shuffle = False,
                    drop_last = True,
                    num_workers=1)

loop = enumerate(train_dataloader)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6,6))


for i, batch in loop:
    img, _, _ = batch
    print(img.size())
    if i > 0 :
        plt.close() 
        raise NotImplementedError
    
    else :    
        frames = list()
        step = 0
        for t in trange(len(img[0])): # iterate over first member only
            ax.imshow(img[0][t], origin="lower", clim=(torch.min(img[0][t]), torch.max(img[0][t])), cmap="coolwarm")
            ax.set_title("[t2m in K] time : +{}h".format(3*t))
            fig.suptitle(f"leadtime {t}")
            # fig.tight_layout()
            frames.append(create_frame(fig))
            step+=1
        frame_one = frames[0]
        frame_one.save(
            '/project/scratch/p200177/DE_371/victorsanchez/results/temporal_gif/' + f"plot_time_step_period_{config.timestep_period}.gif",
            format="GIF",
            append_images=frames,
            save_all=True,
            duration=30*step,
            loop=0,
        )





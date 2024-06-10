import gan.data.dataset_handler_ddp as DSH
from torch.utils.data import DataLoader, Dataset
import numpy as np
from expe_init import get_expe_parameters
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image

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

Dl_train = DSH.ISData_Loader("Train", config)
dataset = DSH.ISDataset(config, Dl_train.dataset_handler_yaml, 'coords', variable_indices=[3], transform=Dl_train.transform())
train_dataloader = DataLoader(dataset = dataset,
                    batch_size = 16,
                    shuffle = False,
                    drop_last = True,
                    num_workers=1)

loop = enumerate(train_dataloader)


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6,6))
frames = list()

for i, batch in loop:
    img, _, _ = batch
    if i > 2 : raise NotImplementedError
    
    else :    
        step = 0
        for t in trange(len(img[0])): # iterate over first member only
            ax.imshow(img[0][t], origin="lower")
            ax.set_title("[t2m in K]")
            # fig.tight_layout()
            frames.append(create_frame(fig))
            step+=1
plt.close()

frame_one = frames[0]
frame_one.save(
    '/scratch/mrmn/sanchezv/project/results/' + "plot.gif",
    format="GIF",
    append_images=frames,
    save_all=True,
    duration=step,
    loop=0,
)



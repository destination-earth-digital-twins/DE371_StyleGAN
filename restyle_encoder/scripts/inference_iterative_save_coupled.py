import os
from argparse import Namespace
from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from restyle_encoder.configs import data_configs
from restyle_encoder.datasets.inference_dataset import InferenceDataset
from options.test_options import TestOptions
from restyle_encoder.models.psp import pSp
from restyle_encoder.models.e4e import e4e
from restyle_encoder.utils.model_utils import ENCODER_TYPES
from restyle_encoder.utils.common import tensor2im
from restyle_encoder.utils.inference_utils import run_on_batch, get_average_image


def run():
    test_config = TestOptions().parse()

    out_path_coupled = os.path.join(test_config.exp_dir, 'inference_coupled')
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_config.checkpoint_path, map_location='cpu')
    config = ckpt['config']
    config.update(vars(test_config))
    config = Namespace(**config)

    if config.encoder_type in ENCODER_TYPES['pSp']:
        net = pSp(config)
    else:
        net = e4e(config)

    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(config.dataset_type))
    dataset_args = data_configs.DATASETS[config.dataset_type]
    transforms_dict = dataset_args['transforms'](config).get_transforms()
    dataset = InferenceDataset(root=config.data_path,
                               transform=transforms_dict['transform_inference'],
                               config=config)
    dataloader = DataLoader(dataset,
                            batch_size=config.test_batch_size,
                            shuffle=False,
                            num_workers=int(config.test_workers),
                            drop_last=False)

    if config.n_images is None:
        config.n_images = len(dataset)

    # get the image corresponding to the latent average
    avg_image = get_average_image(net, config)

    if config.dataset_type == "cars_encode":
        resize_amount = (256, 192) if config.resize_outputs else (512, 384)
    else:
        resize_amount = (256, 256) if config.resize_outputs else (config.output_size, config.output_size)

    global_i = 0
    global_time = []
    for input_batch in tqdm(dataloader):
        if global_i >= config.n_images:
            break

        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            result_batch, result_latents = run_on_batch(input_cuda, net, config, avg_image)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(input_batch.shape[0]):
            results = [tensor2im(result_batch[i][iter_idx]) for iter_idx in range(config.n_iters_per_batch)]
            im_path = dataset.paths[global_i]

            # save step-by-step results side-by-side
            input_im = tensor2im(input_batch[i])
            res = np.array(results[0].resize(resize_amount))
            for idx, result in enumerate(results[1:]):
                res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
            res = np.concatenate([res, input_im.resize(resize_amount)], axis=1)

            Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            global_i += 1

    stats_path = os.path.join(config.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)


if __name__ == '__main__':
    run()

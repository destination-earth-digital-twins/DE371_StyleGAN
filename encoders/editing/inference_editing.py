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

from encoders.configs import data_configs
from encoders.datasets.inference_dataset import InferenceDataset
from editing.latent_editor import LatentEditor
from encoders.models.e4e import e4e
from options.test_options import TestOptions
from encoders.utils.common import tensor2im
from encoders.utils.inference_utils import get_average_image, run_on_batch


def run():
    """
    This script can be used to perform inversion and editing. Please note that this script supports editing using
    only the ReStyle-e4e model and currently supports editing using three edit directions found using InterFaceGAN
    (age, smile, and pose) on the faces domain.
    For performing the edits please provide the arguments `--edit_directions` and `--factor_ranges`. For example,
    setting these values to be `--edit_directions=age,smile,pose` and `--factor_ranges=5,5,5` will use a lambda range
    between -5 and 5 for each of the attributes. These should be comma-separated lists of the same length. You may
    get better results by playing around with the factor ranges for each edit.
    """
    test_config = TestOptions().parse()

    out_path_results = os.path.join(test_config.exp_dir, 'editing_results')
    out_path_coupled = os.path.join(test_config.exp_dir, 'editing_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_config.checkpoint_path, map_location='cpu')
    config = ckpt['config']
    config.update(vars(test_config))
    config = Namespace(**config)
    net = e4e(config)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(config.dataset_type))
    if config.dataset_type != "ffhq_encode":
        raise ValueError("Editing script only supports edits on the faces domain!")
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

    latent_editor = LatentEditor(net.decoder)
    config.edit_directions = config.edit_directions.split(',')
    config.factor_ranges = [int(factor) for factor in config.factor_ranges.split(',')]
    if len(config.edit_directions) != len(config.factor_ranges):
        raise ValueError("Invalid edit directions and factor ranges. Please provide a single factor range for each"
                         f"edit direction. Given: {config.edit_directions} and {config.factor_ranges}")

    avg_image = get_average_image(net, config)

    global_i = 0
    global_time = []
    for input_batch in tqdm(dataloader):
        if global_i >= config.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            result_batch = edit_batch(input_cuda, net, avg_image, latent_editor, config)
            toc = time.time()
            global_time.append(toc - tic)

        resize_amount = (256, 256) if config.resize_outputs else (config.output_size, config.output_size)
        for i in range(input_batch.shape[0]):

            im_path = dataset.paths[global_i]
            results = result_batch[i]

            inversion = results.pop('inversion')
            input_im = tensor2im(input_batch[i])

            all_edit_results = []
            for edit_name, edit_res in results.items():
                res = np.array(input_im.resize(resize_amount))  # set the input image
                res = np.concatenate([res, np.array(inversion.resize(resize_amount))], axis=1)  # set the inversion
                for result in edit_res:
                    res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
                res_im = Image.fromarray(res)
                all_edit_results.append(res_im)

                edit_save_dir = os.path.join(out_path_results, edit_name)
                os.makedirs(edit_save_dir, exist_ok=True)
                res_im.save(os.path.join(edit_save_dir, os.path.basename(im_path)))

            # save final concatenated result if all factor ranges are equal
            if config.factor_ranges.count(config.factor_ranges[0]) == len(config.factor_ranges):
                coupled_res = np.concatenate(all_edit_results, axis=0)
                im_save_path = os.path.join(out_path_coupled, os.path.basename(im_path))
                Image.fromarray(coupled_res).save(im_save_path)

            global_i += 1

    stats_path = os.path.join(config.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)


def edit_batch(inputs, net, avg_image, latent_editor, config):
    y_hat, latents = get_inversions_on_batch(inputs, net, avg_image, config)
    # store all results for each sample, split by the edit direction
    results = {idx: {'inversion': tensor2im(y_hat[idx])} for idx in range(len(inputs))}
    for edit_direction, factor_range in zip(config.edit_directions, config.factor_ranges):
        edit_res = latent_editor.apply_interfacegan(latents=latents,
                                                    direction=edit_direction,
                                                    factor_range=(-1 * factor_range, factor_range))
        # store the results for each sample
        for idx, sample_res in edit_res.items():
            results[idx][edit_direction] = sample_res
    return results


def get_inversions_on_batch(inputs, net, avg_image, config):
    result_batch, result_latents = run_on_batch(inputs, net, config, avg_image)
    # we'll take the final inversion as the inversion to edit
    y_hat = [result_batch[idx][-1] for idx in range(len(result_batch))]
    latents = [torch.from_numpy(result_latents[idx][-1]).cuda() for idx in range(len(result_batch))]
    return y_hat, torch.stack(latents)


if __name__ == '__main__':
    run()
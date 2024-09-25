import numpy as np
import argparse
import perturbation.utils as utils
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--inv_data_dir', type=str, default='/project/scratch/p200177/DE_371/victorsanchez/results/Grand_Ensemble/Inversion/')
parser.add_argument('--gen_data_dir',type = str, default ="/project/scratch/p200177/DE_371/victorsanchez/results/Grand_Ensemble/Perturbation/stochastic_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_False/samples/")
parser.add_argument('--pack_dir',type = str, default ='/project/scratch/p200177/DE_371/victorsanchez/results/Grand_Ensemble/Pack/')
parser.add_argument('--output_dir',type = str, default ='/project/scratch/p200177/DE_371/victorsanchez/results/Grand_Ensemble/Final/')
parser.add_argument('--num_member',type = int, default = 875)
parser.add_argument("--leadtimes", type=utils.str2intlist, default=[6,12,18,24,30,36,42])
parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
parser.add_argument("--inv_step", type=int, default=2000, help='step of inversion to load w')

params = parser.parse_args()

if not os.path.exists(params.output_dir):
    os.makedirs(params.output_dir)
    os.makedirs(params.output_dir+'Pack/')
    os.makedirs(params.output_dir+'Inversion/')
    os.makedirs(params.output_dir+'Gen/')

members = list(range(params.num_member))

for lt in tqdm(params.leadtimes) :
    if not os.path.isfile(params.output_dir + f'Pack/Rsemble_{lt}_875.npy'):
        # Loading Original samples
        Ens_r=utils.collate_R_ensemble(
            data_dir=params.pack_dir,
            members=members,
            lead_time=lt,
            var_indices=params.var_indices
        )
        np.save(params.output_dir + f'Pack/Rsemble_{lt}_875.npy', Ens_r)

    if not os.path.isfile(params.output_dir + f'Inversion/invertFsemble_{lt}_875.npy'):
        # Loading Inverted samples
        inv_ens=utils.collate_inv_ensemble(
            data_dir=params.inv_data_dir,
            members=members,
            lead_time=lt,
            var_indices=params.var_indices,
            inv_step=params.inv_step
        )
        np.save(params.output_dir + f'Inversion/invertFsemble_{lt}_875.npy', inv_ens)

    # # Loading Generated samples
    # gen_ens = utils.collate_gen_ensemble(
    #     data_dir=params.gen_data_dir,
    #     members=members,
    #     lead_time=lt,
    #     var_indices=params.var_indices,
    #     inv_step=params.inv_step
    # )
    # np.save(params.output_dir + f'Gen/genFsemble_{lt}_{params.inv_step}_875.npy', gen_ens)
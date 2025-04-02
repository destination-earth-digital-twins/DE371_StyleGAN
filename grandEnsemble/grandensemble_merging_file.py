import numpy as np
import argparse
import utils.utils as utils
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--inv_data_dir', type=str, default='')
parser.add_argument('--gen_data_dir',type = str, default ="")
parser.add_argument('--pack_dir',type = str, default ='')
parser.add_argument('--output_dir',type = str, default ='')
parser.add_argument('--num_member',type = int, default = 875)
parser.add_argument("--leadtimes", type=utils.str2intlist, default=[6,12,18,24,30,36,42])
parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
parser.add_argument("--inv_step", type=int, default=250, help='step of inversion to load w')

params = parser.parse_args()


os.makedirs(params.output_dir, exist_ok=True)
os.makedirs(params.output_dir+'Pack/', exist_ok=True)
os.makedirs(params.output_dir+'Inversion/', exist_ok=True)
os.makedirs(params.output_dir+'Gen/', exist_ok=True)

members = list(range(params.num_member))

for lt in tqdm(params.leadtimes) :
    if not os.path.isfile(params.output_dir + f'Pack/Rsemble_{lt}_875.npy'):
        # Loading Original samples
        print('Merging Rsemble files')
        Ens_r=utils.collate_R_ensemble(
            data_dir=params.pack_dir,
            members=members,
            lead_time=lt,
            var_indices=params.var_indices
        )
        #  DENORM DATA IF NECESSARY
        np.save(params.output_dir + f'Pack/Rsemble_{lt}_875.npy', Ens_r)

    if not os.path.isfile(params.output_dir + f'Inversion/invertFsemble_{lt}_875_{params.inv_step}.npy'):
        # Loading Inverted samples
        print('Merging invertFsemble files')
        inv_ens=utils.collate_inv_ensemble(
            data_dir=params.inv_data_dir,
            members=members,
            lead_time=lt,
            var_indices=params.var_indices,
            inv_step=params.inv_step
        )
        np.save(params.output_dir + f'Inversion/genFsemble__{lt}_875_{params.inv_step}.npy', inv_ens)

    if not os.path.isfile(params.output_dir + f'Gen/genFsemble_{lt}_875_{params.inv_step}.npy'):
        # Loading Generated samples
        print('Merging genFsemble files')
        gen_ens = utils.collate_gen_ensemble(
            data_dir=params.gen_data_dir,
            members=members,
            lead_time=lt,
            var_indices=params.var_indices,
            inv_step=params.inv_step
        )
        np.save(params.output_dir + f'Gen/genFsemble_{lt}_875_{params.inv_step}.npy', gen_ens)
import torch
import perturbation.inversion as inversion_test_bug
import perturbation.inversion_wo_noise as inv_wonoise
import argparse
from gan.model.stylegan2 import Generator
import os
import numpy as np
from time import perf_counter
from collections import OrderedDict
import perturbation.utils as utils
import pickle

torch.manual_seed(42) #reproducibility of runs

if __name__=="__main__" :


    
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################

    parser.add_argument('--ckpt_dir', type = str, 
                        default ='/project/scratch/p200177/DE_371/angeliquebonamy/stylegan2_stylegan_dom_256_lat-dim_512_bs_8_0.002_0.002_ch-mul_2_vars_rr_u_v_t2m_noise_True/model/222000.pt')
    parser.add_argument('--real_data_dir', type = str, 
                        default ='/project/scratch/p200177/DE_371/angeliquebonamy/data_basile_inv/samples_AROME_for_AE_1/')
    parser.add_argument('--output_dir',type = str, 
                        default ='/project/scratch/p200177/DE_371/angeliquebonamy/results/inversion/')
    parser.add_argument("--pack_dir", type=str, default = '/project/scratch/p200177/DE_371/angeliquebonamy/results/pack/') # storing "packed" (normalized) real data
    parser.add_argument("--stat_dir", type=str, default = '//project/scratch/p200177/DE_371/angeliquebonamy/data_basile_inv/samples_AROME_for_AE_1/stat/stat_file/')
    parser.add_argument('--min_file', type=str, default='min_rr_log.npy')
    parser.add_argument('--max_file', type=str, default='max_rr_log.npy')
    parser.add_argument('--use_noise',action="store_true")
    parser.add_argument('--device', type=str, default='cuda:0')
    ############################ INVERSION PARAMETERS #################    

    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float, 
        default=0.25,
        help="duration of the learning rate decay",
    )
    
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    
  
    parser.add_argument(
        "--noise", type=float, default=0.005, help="strength of the noise level"
    )
    
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    
    parser.add_argument("--invstep", type=int, default=1000, help="optimize iterations")
    parser.add_argument("--var_indices", type=utils.str2intlist, default=[0,1,2,3])
    parser.add_argument("--Shape", type=tuple, default=(4,256,256), help='size of the samples')
    
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=10e5,
        help="weight of the noise regularization (inversion)",
    )

    parser.add_argument('--loss', type=str, default='mse', choices = ['mse', 'mae'])
    parser.add_argument("--loss_intens", type=float, default=1.0, help="weight of the pixel loss")

    parser.add_argument("--inv_checkpoints", type=utils.str2intlist, default=[200,400,600,800,1000])

    ########################## CONTROL of Data to invert ######################

    params = parser.parse_args()


    #print(params.inv_checkpoints, type(params.inv_checkpoints[0]))
    assert type(params.inv_checkpoints[0])==int
    ################## loading data to invert ##

    classes = pickle.load(open(params.real_data_dir + 'class_samples.p','rb'))
    scenarii = classes.keys()

    Mins = np.load(f'{params.stat_dir}{params.min_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)

    Maxs = np.load(f'{params.stat_dir}{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)

    ################ loading network #################

    device = params.device if torch.cuda.is_available() else 'cpu'

    G = Generator(params.Shape[1], 512,n_mlp=8,nb_var=params.Shape[0],var_rr=True,use_noise=params.use_noise,tanh_output=True)


    #print('###########################################"##################################################################################################################')
    ckpt = torch.load(params.ckpt_dir, map_location='cpu')['g_ema']

    if 'module' in list(ckpt.items())[0][0]: #juglling with Pytorch versioning and different module packaging
        ckpt_adapt = OrderedDict()
        for k in ckpt.keys():
            k0 = k[7:]
            ckpt_adapt[k0] = ckpt[k]
        G.load_state_dict(ckpt_adapt)


    else:
        G.load_state_dict(ckpt_dic)
    G.eval()
    G = G.to(device)

    ################### producing latent mean #######

    if not os.path.exists(f'{params.output_dir}latent_mean.npy'):

        latent_z = torch.empty(10000, 512).normal_().to(device)
        with torch.no_grad():
            w = G.style(latent_z)

        latent_mean = w.mean(dim=0).detach().cpu()

        np.save(f'{params.output_dir}latent_mean.npy',latent_mean.numpy())
    else : 

        lm = np.load(f'{params.output_dir}latent_mean.npy').astype(np.float32)
        latent_mean = torch.tensor(lm, dtype = torch.float32)


    #################### main loop ##################

    for scenario in scenarii:
        samples = torch.tensor(classes[scenario])
        samples = torch.split(samples,[16,16,16,2],dim=0)
        for batch_idx, batch in enumerate(samples[:-1]):
            print(scenario, batch_idx)
            params.date_index, params.lt_index = scenario, batch_idx
            if not os.path.exists(params.output_dir +f'w_{scenario}_{batch_idx}_1000.npy'): #checking for already teer

                batch[:,0] =  -1.0 + 2 * (torch.log(1 + batch[:,0]) - Mins[:,0]) / (Maxs[:,0] - Mins[:,0])
                batch[:,1:] = -1.0 + 2 * (batch[:,1:] - Mins[:,1:]) / (Maxs[:,1:] - Mins[:,1:]) 
                np.save(params.pack_dir+f'Rsemble_{scenario}_{batch_idx}.npy', batch.numpy().astype(np.float32))
                if params.use_noise:
                    inv.optimize(batch, G, latent_mean, device, params)
                else:
                    inv_wonoise.optimize(batch, G, latent_mean, device, params)

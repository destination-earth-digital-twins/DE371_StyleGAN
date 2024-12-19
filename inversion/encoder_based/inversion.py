import torch
import numpy as np
from encoders.utils import common, train_utils
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import utils.utils as utils
from time import time
from inversion.encoder_based.encoder_utils import log_images_diff


def inversion_restyle(params, network, Ens_r):
    y_hat, latent = None, None

    y_hats = {idx: [] for idx in range(Ens_r.shape[0])}
    mem_cuda = torch.cuda.memory_allocated(device=params.device)
    # print('memory_allocated {}'.format(humanbytes(mem_cuda)))
    # get the image corresponding to the latent average
    avg_sample = network(
        network.latent_avg.unsqueeze(0),
        input_code=True,
        randomize_noise=False,
        return_latents=False,
        average_code=True)[0]
    with torch.no_grad():
        avg_sample = avg_sample.to(params.device).float()                
        Ens_r = Ens_r.to(params.device)
        mem_cuda = torch.cuda.memory_allocated(device=params.device)
        # print('memory_allocated {}'.format(humanbytes(mem_cuda)))
        t0 = time()
        # Restyle-Encoder Loop
        for iter in range(params.n_iters_per_batch):
            print('iter : ', iter)
            if iter == 0:
                avg_image_for_batch = avg_sample.unsqueeze(0).repeat(Ens_r.shape[0], 1, 1, 1)
                x_input = torch.cat([Ens_r, avg_image_for_batch], dim=1)
            else:
                x_input = torch.cat([Ens_r, y_hat], dim=1)
    
            y_hat, latent = network.forward(x_input, latent=latent, return_latents=True)
            
            for idx in range(Ens_r.shape[0]):
                y_hats[idx].append([y_hat[idx]])

            if iter+1 in params.n_iters_per_batch_checkpoint :
                print("--saving inverted samples :", params.output_dir+'w_{}_{}_{}.npy'.format(params.date_index,params.lt_index, iter+1))
                np.save(params.output_dir+'w_{}_{}_{}.npy'.format(params.date_index,params.lt_index, iter+1),latent.cpu().detach().numpy())
                np.save(params.output_dir+'invertFsemble_{}_{}_{}.npy'.format(params.date_index,params.lt_index, iter+1),y_hat.cpu().detach().numpy())
                if params.plot_checkpoint:
                    print("--plotting inverted samples :", params.output_dir+'w_{}_{}.npy'.format(params.date_index,params.lt_index))
                    log_images_diff(
                        config=params,
                        x=Ens_r,
                        y_hat=y_hats,
                        iter=iter+1
                    )
        print('Time taken for inversion :', time()-t0)

    return y_hats
    


def inversion_psp_e4e(params, network, Ens_r):

    y_hat, latent = None, None

    mem_cuda = torch.cuda.memory_allocated(device=params.device)
    # print('memory_allocated {}'.format(humanbytes(mem_cuda)))

    with torch.no_grad():             
        Ens_r = Ens_r.to(params.device)
        mem_cuda = torch.cuda.memory_allocated(device=params.device)

        t0 = time()
        y_hat, latent = network.forward(Ens_r, return_latents=True)
        print('Time taken for inversion :', time()-t0)

        np.save(params.output_dir+'w_{}_{}'.format(params.date_index,params.lt_index),latent.cpu().detach().numpy())
        np.save(params.output_dir+'invertFsemble_{}_{}'.format(params.date_index,params.lt_index),y_hat.cpu().detach().numpy())

    return y_hat
    

def inversion_inDomain(params, network, Ens_r):

    y_hat, latent = None, None

    mem_cuda = torch.cuda.memory_allocated(device=params.device)
    # print('memory_allocated {}'.format(humanbytes(mem_cuda)))

    with torch.no_grad():             
        Ens_r = Ens_r.to(params.device)
        mem_cuda = torch.cuda.memory_allocated(device=params.device)
        t0 = time()
        y_hat, latent, discrim_out_real, discrim_out_fake = network.forward(Ens_r, return_latents=True)
        print('Time taken for inversion :', time()-t0)

        np.save(params.output_dir+'w_{}_{}.npy'.format(params.date_index,params.lt_index),latent.cpu().detach().numpy())
        np.save(params.output_dir+'invertFsemble_{}_{}.npy'.format(params.date_index,params.lt_index),y_hat.cpu().detach().numpy())
        


    return y_hat
    
def inversion_featureStyle(params, network, Ens_r):

    y_hat, latent = None, None

    mem_cuda = torch.cuda.memory_allocated(device=params.device)
    # print('memory_allocated {}'.format(humanbytes(mem_cuda)))

    with torch.no_grad():             
        Ens_r = Ens_r.to(params.device)
        mem_cuda = torch.cuda.memory_allocated(device=params.device)
        t0 = time()
        concat_img, _, _, _, y_hat, latent = network.forward(Ens_r, feature_scale=1, train=False, return_latent=True)
        print('Time taken for inversion :', time()-t0)
        b = concat_img.size(0)//2  
        
        np.save(params.output_dir+'w_{}_{}.npy'.format(params.date_index,params.lt_index),latent.cpu().detach().numpy())
        np.save(params.output_dir+'invertFsemble_{}_{}.npy'.format(params.date_index,params.lt_index),y_hat[b:].cpu().detach().numpy())
        
        

    return y_hat[b:]
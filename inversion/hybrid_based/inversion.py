import torch
import matplotlib
matplotlib.use('Agg')
from generate_sample import humanbytes



def init_latent_restyle(params, network, Ens_r):
    y_hat, latent = None, None
    # latent_complete = torch.empty((Ens_r.shape[0], 14, 512))
    # y_hat_complete = torch.empty(Ens_r.shape)
    y_hats = {idx: [] for idx in range(Ens_r.shape[0])}
    mem_cuda = torch.cuda.memory_allocated(device=params.device)
    print('memory_allocated {}'.format(humanbytes(mem_cuda)))
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

    return latent
    


def init_latent_psp_e4e(params, network, Ens_r):

    y_hat, latent = None, None

    mem_cuda = torch.cuda.memory_allocated(device=params.device)
    print('memory_allocated {}'.format(humanbytes(mem_cuda)))

    with torch.no_grad():             
        Ens_r = Ens_r.to(params.device)
        y_hat, latent = network.forward(Ens_r, return_latents=True)

    return latent
    

def init_latent_inDomain(params, network, Ens_r):

    y_hat, latent = None, None

    with torch.no_grad():             
        Ens_r = Ens_r.to(params.device)
        y_hat, latent, discrim_out_real, discrim_out_fake = network.forward(Ens_r, return_latents=True)

    return latent
    
def init_latent_featureStyle(params, network, Ens_r):

    y_hat, latent = None, None

    with torch.no_grad():             
        Ens_r = Ens_r.to(params.device)
        concat_img, _, _, _, y_hat, latent = network.forward(Ens_r, feature_scale=1, train=False, return_latent=True)

    return latent
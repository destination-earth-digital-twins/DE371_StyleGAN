from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

color_dict = ['viridis', 'viridis', 'coolwarm']
var_list = ['rr',  'u', 'v', 't2m', 'orog', 'z500', 't850', 'tpw850']

def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))

###############################################################################
        ###############################################################
###############################################################################


def vis_samples_iterative(hooks_dict, fig, gs, i, n_vars, n_outputs):

    vmin = []
    vmax = []
    for j in range(n_vars):
         ax = fig.add_subplot(gs[n_outputs * i,j])
         vmin.append(hooks_dict['input'][j,:,:].min())
         vmax.append(hooks_dict['input'][j,:,:].max())
         ax.set_title(f"{var_list[j+1]} real")
         im = ax.imshow(hooks_dict['input'][j,:,:], origin ='lower', cmap = color_dict[j], vmin = vmin[-1], vmax = vmax[-1])
         fig.colorbar(im, shrink=0.5)
    
    
    # plt.title('Input sample {}'.format(str(i)))
    # fig.add_subplot(gs[i, 1])
    # plt.imshow(hooks_dict['target'])
    # plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']), float(hooks_dict['diff_target'])))

    
    for idx, output_idx in enumerate(range(len(hooks_dict['output']) - 1, -1, -1)):

        output = hooks_dict['output'][output_idx]
        
        for j in range(n_vars):
            ax = fig.add_subplot(gs[n_outputs * i + 1 + idx, j])
            ax.set_title(f"{var_list[j+1]} inv | Iter {str(n_outputs - idx)} ")
            im = ax.imshow(output[0][j,:,:], origin ='lower', cmap = color_dict[j], vmin = vmin[j], vmax = vmax[j])
            fig.colorbar(im, shrink=0.5)
            #if j==2 :
        # plt.title('Output sample {}, Iter  {}'.format(str(i), str(n_outputs - idx)))

def vis_samples(log_hooks, n_vars):
    
    display_count = len(log_hooks)

    n_outputs = len(log_hooks[0]['output']) if type(log_hooks[0]['output']) == list else 1 # n_outputs is the number of iteration steps (?)

    fig = plt.figure(figsize=(6 + 1.5 * n_vars, 2 * (n_outputs +1) * display_count))
    gs = fig.add_gridspec(display_count * (n_outputs + 1),  ( n_vars ))
	
    for i in range(display_count):
        
        hooks_dict = log_hooks[i]
        
        vis_samples_iterative(hooks_dict, fig, gs, i, n_vars, n_outputs)
        if i == 0 :
            fig.suptitle('Result of Encoder Inversion over Iterations', fontsize=20)
    plt.tight_layout()
    
    return fig

def vis_samples_diff(log_hooks, n_vars, single=False):
    
    display_count = len(log_hooks)

    fig = plt.figure(figsize=(15,15))
    gs = fig.add_gridspec(display_count * 3,  (n_vars))
	
    for i in range(display_count):
        
        hooks_dict = log_hooks[i]
        for j in range(n_vars):
            real_sample = hooks_dict['input'][j,:,:]
            if single :
                inv_sample = hooks_dict['output'][-1][j,:,:]
            else :
                inv_sample = hooks_dict['output'][-1][0][j,:,:]
            diff = real_sample - inv_sample
            vmin=real_sample.min()
            vmax=real_sample.max()
            ax = fig.add_subplot(gs[i,j])
            ax.set_title(f"{var_list[j+1]} real")
            im = ax.imshow(real_sample, origin ='lower', cmap = color_dict[j], vmin = vmin, vmax = vmax)
            fig.colorbar(im, shrink=0.5)

            ax = fig.add_subplot(gs[i + 1, j])
            ax.set_title(f"{var_list[j+1]} inv")
            im = ax.imshow(inv_sample, origin ='lower', cmap = color_dict[j], vmin = vmin, vmax = vmax)
            fig.colorbar(im, shrink=0.5)

            ax = fig.add_subplot(gs[i + 2, j])
            ax.set_title(f"{var_list[j+1]} diff")
            im = ax.imshow(diff, origin ='lower', cmap="RdYlGn", vmin = -0.1, vmax = 0.1)
            fig.colorbar(im, shrink=0.5)
        if i == 0 :
            fig.suptitle('Result of Encoder Inversion', fontsize=20)
        
    plt.tight_layout()
    
    return fig

def numpyfy(var, normalize=False, Mean=None, Max=None, save=False, name=None):
    
    var = var.cpu().detach().numpy()
    
    if normalize :
        
        assert (Mean is not None) and (Max is not None)
        
        var = var * Max + Mean
    
    if save :
        
        assert name is not None
        np.save(name+'.npy', var)
    
    return var
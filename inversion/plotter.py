import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4, 'z500': 5, 't850': 6, 'tpw850': 7}


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

def online_inv_plot(
          packsample,
          invsample,
          crop=[0,-1,0,-1],
          mem_idx=0,
          figtitle=" ",
          figname="inv.png",
          var_names=['u','v','t2m'],
          dict_var={'u': 0, 'v': 1, 't2m': 2},
          colormap_var=['viridis','viridis','coolwarm'],
          savefig=True
          ):
        
        fig, ax = plt.subplots(figsize=(15,5*len(var_names)), nrows=3, ncols=len(var_names))
        
        for id, var in enumerate(var_names):

            var_id = dict_var[var]
            vmin = np.min([np.min(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]])])
            vmax = np.min([np.max(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]])])

            
            ax[0][id].set_title(f"{var} real")
            im = ax[0][id].imshow(packsample[mem_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap=colormap_var[id])
            fig.colorbar(im, ax=ax[0][id], shrink=0.5)

            
            im = ax[1][id].imshow(invsample[mem_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower",  cmap=colormap_var[id])
            ax[1][id].set_title(f"{var} inv")
            fig.colorbar(im, ax=ax[1][id], shrink=0.5)

            
            diff = packsample[mem_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]] - invsample[mem_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]]
            im = ax[2][id].imshow(diff, origin="lower", cmap="RdYlGn")
            im.set_clim(-0.1,0.1)
            ax[2][id].set_title("diff")
            fig.colorbar(im, ax=ax[2][id], shrink=0.5)

        fig.suptitle(figtitle)
        fig.tight_layout()
        if savefig:
            try:
                fig.savefig(figname, dpi=100)
            except Exception:
                print(f"unable to save figure: {figname}")
            plt.close()
        else :
            plt.close()
            return fig


def online_inv_temporal_plot(
          packsample,
          invsample, crop=[0,-1,0,-1],
          mem_idx=0,
          nb_timesteps=15,
          var_names=['u','v','t2m'],
          figtitle=" ",
          figname="inv.png",
          colormap_var=['viridis','viridis','coolwarm']

          ):
        nb_var=len(var_names)
        for id, var_label in enumerate(var_names):
            fig, ax = plt.subplots(nrows=3, ncols=nb_timesteps)
            for t in range(nb_timesteps):
                vmin = np.min([np.min(packsample[:,id+nb_var*t,crop[0]:crop[1],crop[2]:crop[3]])])
                vmax = np.min([np.max(packsample[:,id+nb_var*t,crop[0]:crop[1],crop[2]:crop[3]])])
                im=ax[0][t].imshow(packsample[mem_idx,id+nb_var*t,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower",cmap=colormap_var[id])
                ax[0][t].set_title(f"{var_label} real")
                if t == nb_timesteps-1:
                    fig.colorbar(im, ax=ax[0][t], shrink=0.5)

                im=ax[1][t].imshow(invsample[mem_idx,id+nb_var*t,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower",cmap=colormap_var[id])
                ax[1][t].set_title(f"{var_label} inv")
                if t == nb_timesteps-1:
                    fig.colorbar(im, ax=ax[1][t], shrink=0.5)
                
                diff = packsample[mem_idx,id+nb_var*t,crop[0]:crop[1],crop[2]:crop[3]] - invsample[mem_idx,id+nb_var*t,crop[0]:crop[1],crop[2]:crop[3]]
                im=ax[2][t].imshow(diff, origin="lower", cmap="RdYlGn")
                im.set_clim(-0.1,0.1)
                ax[2][t].set_title("diff")
                if t == nb_timesteps-1:
                    fig.colorbar(im, ax=ax[2][t], shrink=0.5)

            fig.suptitle(figtitle+f"_{var_label}")
            fig.tight_layout()
            try:
                fig.savefig(figname, dpi=100)
            except Exception:
                print(f"unable to save figure: {figname}")
            plt.close()
        return
    

def online_pert_plot(
          packsample, 
          invsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          mem_idx=0, 
          mem_pert_idx=0,figtitle=" ", 
          figname="inv.png",  
          var_names=['u','v','t2m'], 
          dict_var={'u': 0, 'v': 1, 't2m': 2},
          colormap_var=['viridis','viridis','coolwarm']
          ):

        fig, ax = plt.subplots(figsize=(15,5*len(var_names)), nrows=3, ncols=len(var_names))
        for id, var in enumerate(var_names):
            var_id = dict_var[var]

            vmin = np.min([np.min(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]])])
            vmax = np.min([np.max(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]])])

            ax[0][id].set_title(f"{var} real")
            im = ax[0][id].imshow(packsample[mem_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap=colormap_var[id])
            fig.colorbar(im, ax=ax[0][id], shrink=0.5)

            ax[1][id].set_title(f"{var} inv")
            im = ax[1][id].imshow(invsample[mem_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap=colormap_var[id])
            fig.colorbar(im, ax=ax[1][id], shrink=0.5)

            ax[2][id].set_title(f"{var} perturbated")
            im = ax[2][id].imshow(pert_sample[mem_pert_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap=colormap_var[id])
            fig.colorbar(im, ax=ax[2][id], shrink=0.5)

        fig.suptitle(figtitle)
        fig.tight_layout()
        try:
            fig.savefig(figname, dpi=100)
        except Exception:
            print(f"unable to save figure: {figname}")
        plt.close()
        return


def online_pert_diff_plot(
          invsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          mem_idx=0, 
          mem_pert_idx=0,figtitle=" ", 
          figname="inv.png",  
          var_names=['u','v','t2m'], 
          dict_var={'u': 0, 'v': 1, 't2m': 2},
          colormap_var=['viridis','viridis','coolwarm']
          ):

        fig, ax = plt.subplots(figsize=(15,5*len(var_names)), nrows=3, ncols=len(var_names))
        for id, var in enumerate(var_names):
            var_id = dict_var[var]

            vmin = np.min([np.min(invsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]])])
            vmax = np.min([np.max(invsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]])])

            ax[0][id].set_title(f"{var} inv")
            im = ax[0][id].imshow(invsample[mem_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap=colormap_var[id])
            fig.colorbar(im, ax=ax[0][id], shrink=0.5)

            ax[1][id].set_title(f"{var} perturbated")
            im = ax[1][id].imshow(pert_sample[mem_pert_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap=colormap_var[id])
            fig.colorbar(im, ax=ax[1][id], shrink=0.5)

            diff = pert_sample[mem_pert_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]] - invsample[mem_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]]
            ax[2][id].set_title(f"{var} diff")
            im = ax[2][id].imshow(diff, clim=(vmin, vmax), origin="lower", cmap="RdYlGn")
            im.set_clim(-0.1,0.1)
            fig.colorbar(im, ax=ax[2][id], shrink=0.5)

        fig.suptitle(figtitle)
        fig.tight_layout()
        try:
            fig.savefig(figname, dpi=100)
        except Exception:
            print(f"unable to save figure: {figname}")
        plt.close()
        return
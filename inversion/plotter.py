import numpy as np
import matplotlib.pyplot as plt


def online_inv_plot(packsample, invsample, crop=[0,-1,0,-1], mem_idx=0, figtitle=" ", figname="inv.png"):

        fig = plt.figure(figsize=(15,8))

        vmin = np.min([np.min(packsample[:,0,crop[0]:crop[1],crop[2]:crop[3]])])
        vmax = np.min([np.max(packsample[:,0,crop[0]:crop[1],crop[2]:crop[3]])])
        ax = fig.add_subplot(231)
        ax.set_title("u real")
        im = ax.imshow(packsample[mem_idx,0,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
        fig.colorbar(im, shrink=0.5)
        ax = fig.add_subplot(234)
        im = ax.imshow(invsample[mem_idx,0,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
        ax.set_title("u inv")
        fig.colorbar(im, shrink=0.5)

        vmin = np.min([np.min(packsample[:,1,crop[0]:crop[1],crop[2]:crop[3]])])
        vmax = np.min([np.max(packsample[:,1,crop[0]:crop[1],crop[2]:crop[3]])])
        ax = fig.add_subplot(232)
        ax.set_title("v real")
        im = ax.imshow(packsample[mem_idx,1,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
        fig.colorbar(im, shrink=0.5)
        ax = fig.add_subplot(235)
        im = ax.imshow(invsample[mem_idx,1,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
        ax.set_title("v inv")
        fig.colorbar(im, shrink=0.5)

        vmin = np.min([np.min(packsample[:,2,crop[0]:crop[1],crop[2]:crop[3]])])
        vmax = np.min([np.max(packsample[:,2,crop[0]:crop[1],crop[2]:crop[3]])])
        ax = fig.add_subplot(233)
        ax.set_title("t2m real")
        im = ax.imshow(packsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
        fig.colorbar(im, shrink=0.5)
        ax = fig.add_subplot(236)
        im = ax.imshow(invsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
        ax.set_title("t2m inv")
        fig.colorbar(im, shrink=0.5)

        fig.suptitle(figtitle)
        fig.tight_layout()
        try:
            fig.savefig(figname, dpi=100)
        except Exception:
            print(f"unable to save figure: {figname}")
        plt.close()
        return


def online_inv_plot_2(packsample, invsample, crop=[0,-1,0,-1], mem_idx=0, figtitle=" ", figname="inv.png"):

        fig = plt.figure(figsize=(15,15))

        #### u
        vmin = np.min([np.min(packsample[:,0,crop[0]:crop[1],crop[2]:crop[3]])])
        vmax = np.min([np.max(packsample[:,0,crop[0]:crop[1],crop[2]:crop[3]])])

        ax = fig.add_subplot(331)
        ax.set_title("u real")
        im = ax.imshow(packsample[mem_idx,0,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
        fig.colorbar(im, shrink=0.5)

        ax = fig.add_subplot(334)
        im = ax.imshow(invsample[mem_idx,0,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
        ax.set_title("u inv")
        fig.colorbar(im, shrink=0.5)

        ax = fig.add_subplot(337)
        diff = packsample[mem_idx,0,crop[0]:crop[1],crop[2]:crop[3]] - invsample[mem_idx,0,crop[0]:crop[1],crop[2]:crop[3]]
        im = ax.imshow(diff, origin="lower", cmap="RdYlGn")
        im.set_clim(-0.1,0.1)
        ax.set_title("diff")
        fig.colorbar(im, shrink=0.5)


        #### v
        vmin = np.min([np.min(packsample[:,1,crop[0]:crop[1],crop[2]:crop[3]])])
        vmax = np.min([np.max(packsample[:,1,crop[0]:crop[1],crop[2]:crop[3]])])

        ax = fig.add_subplot(332)
        ax.set_title("v real")
        im = ax.imshow(packsample[mem_idx,1,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
        fig.colorbar(im, shrink=0.5)

        ax = fig.add_subplot(335)
        im = ax.imshow(invsample[mem_idx,1,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
        ax.set_title("v inv")
        fig.colorbar(im, shrink=0.5)

        ax = fig.add_subplot(338)
        diff = packsample[mem_idx,1,crop[0]:crop[1],crop[2]:crop[3]] - invsample[mem_idx,1,crop[0]:crop[1],crop[2]:crop[3]]
        im = ax.imshow(diff, origin="lower", cmap="RdYlGn")
        im.set_clim(-0.1,0.1)
        ax.set_title("diff")
        fig.colorbar(im, shrink=0.5)


        #### t2m
        vmin = np.min([np.min(packsample[:,2,crop[0]:crop[1],crop[2]:crop[3]])])
        vmax = np.min([np.max(packsample[:,2,crop[0]:crop[1],crop[2]:crop[3]])])

        ax = fig.add_subplot(333)
        ax.set_title("t2m real")
        im = ax.imshow(packsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
        fig.colorbar(im, shrink=0.5)

        ax = fig.add_subplot(336)
        im = ax.imshow(invsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
        ax.set_title("t2m inv")
        fig.colorbar(im, shrink=0.5)

        ax = fig.add_subplot(339)
        diff = packsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]] - invsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]]
        im = ax.imshow(diff, origin="lower", cmap="RdYlGn")
        im.set_clim(-0.1,0.1)
        ax.set_title("diff")
        fig.colorbar(im, shrink=0.5)

        fig.suptitle(figtitle)
        fig.tight_layout()
        try:
            fig.savefig(figname, dpi=100)
        except Exception:
            print(f"unable to save figure: {figname}")
        plt.close()
        return


def online_inv_temporal_plot(packsample, invsample, crop=[0,-1,0,-1], mem_idx=0, nb_timesteps=15, var_names=['u','v','t2m'], figtitle=" ", figname="inv.png"):
        #TODO : Test tha function
        nb_var=len(var_names)
        for id, var_label in enumerate(var_names):
            fig, ax = plt.subplots(nrows=3, ncols=nb_timesteps)
            for t in range(nb_timesteps):
                vmin = np.min([np.min(packsample[:,id+nb_var*t,crop[0]:crop[1],crop[2]:crop[3]])])
                vmax = np.min([np.max(packsample[:,id+nb_var*t,crop[0]:crop[1],crop[2]:crop[3]])])
                im=ax[0][t].imshow(packsample[mem_idx,id+nb_var*t,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
                ax[0][t].set_title(f"{var_label} real")
                if t == nb_timesteps-1:
                    fig.colorbar(im, shrink=0.5)

                im=ax[1][t].imshow(invsample[mem_idx,id+nb_var*t,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
                ax[1][t].set_title(f"{var_label} inv")
                if t == nb_timesteps-1:
                    fig.colorbar(im, shrink=0.5)
                
                diff = packsample[mem_idx,id+nb_var*t,crop[0]:crop[1],crop[2]:crop[3]] - invsample[mem_idx,id+nb_var*t,crop[0]:crop[1],crop[2]:crop[3]]
                im=ax[2][t].imshow(diff, origin="lower", cmap="RdYlGn")
                im.set_clim(-0.1,0.1)
                ax[2][t].set_title("diff")
                if t == nb_timesteps-1:
                    fig.colorbar(im, shrink=0.5)

            fig.suptitle(figtitle+f"_{var_label}")
            fig.tight_layout()
            try:
                fig.savefig(figname, dpi=100)
            except Exception:
                print(f"unable to save figure: {figname}")
            plt.close()
        return
    

def online_pert_plot(packsample, invsample, pert_sample, crop=[0,-1,0,-1], mem_idx=0, figtitle=" ", figname="inv.png"):

        fig = plt.figure(figsize=(15,15))
        mem_pert_idx = mem_idx # np.random.randint(0, len(pert_sample)-1)
        #### u
        vmin = np.min([np.min(packsample[:,0,crop[0]:crop[1],crop[2]:crop[3]])])
        vmax = np.min([np.max(packsample[:,0,crop[0]:crop[1],crop[2]:crop[3]])])

        ax = fig.add_subplot(331)
        ax.set_title("u real")
        im = ax.imshow(packsample[mem_idx,0,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
        fig.colorbar(im, shrink=0.5)

        ax = fig.add_subplot(334)
        im = ax.imshow(invsample[mem_idx,0,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
        ax.set_title("u inv")
        fig.colorbar(im, shrink=0.5)

        ax = fig.add_subplot(337)
        ax.set_title("u perturbated")
        im = ax.imshow(pert_sample[mem_pert_idx,0,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
        fig.colorbar(im, shrink=0.5)


        #### v
        vmin = np.min([np.min(packsample[:,1,crop[0]:crop[1],crop[2]:crop[3]])])
        vmax = np.min([np.max(packsample[:,1,crop[0]:crop[1],crop[2]:crop[3]])])

        ax = fig.add_subplot(332)
        ax.set_title("v real")
        im = ax.imshow(packsample[mem_idx,1,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
        fig.colorbar(im, shrink=0.5)

        ax = fig.add_subplot(335)
        im = ax.imshow(invsample[mem_idx,1,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
        ax.set_title("v inv")
        fig.colorbar(im, shrink=0.5)

        ax = fig.add_subplot(338)
        ax.set_title("v perturbated")
        im = ax.imshow(pert_sample[mem_pert_idx,1,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
        fig.colorbar(im, shrink=0.5)


        #### t2m
        vmin = np.min([np.min(packsample[:,2,crop[0]:crop[1],crop[2]:crop[3]])])
        vmax = np.min([np.max(packsample[:,2,crop[0]:crop[1],crop[2]:crop[3]])])

        ax = fig.add_subplot(333)
        im = ax.imshow(packsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
        ax.set_title("t2m real")
        fig.colorbar(im, shrink=0.5)

        ax = fig.add_subplot(336)
        im = ax.imshow(invsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
        ax.set_title("t2m inv")
        fig.colorbar(im, shrink=0.5)

        ax = fig.add_subplot(339)
        ax.set_title("t2m perturbated")
        im = ax.imshow(pert_sample[mem_pert_idx,2,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
        fig.colorbar(im, shrink=0.5)

        fig.suptitle(figtitle)
        fig.tight_layout()
        try:
            fig.savefig(figname, dpi=100)
        except Exception:
            print(f"unable to save figure: {figname}")
        plt.close()
        return
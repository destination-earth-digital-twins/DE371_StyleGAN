# import numpy as np
# import matplotlib.pyplot as plt


# def online_inv_plot(packsample, invsample, crop=[0,-1,0,-1], mem_idx=0, figtitle=" ", figname="inv.png"):

#         fig = plt.figure(figsize=(15,8))

#         vmin = np.min([np.min(packsample[:,0,crop[0]:crop[1],crop[2]:crop[3]])])
#         vmax = np.min([np.max(packsample[:,0,crop[0]:crop[1],crop[2]:crop[3]])])
#         ax = fig.add_subplot(231)
#         ax.set_title("u real")
#         im = ax.imshow(packsample[mem_idx,0,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
#         fig.colorbar(im, shrink=0.5)
#         ax = fig.add_subplot(234)
#         im = ax.imshow(invsample[mem_idx,0,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
#         ax.set_title("u inv")
#         fig.colorbar(im, shrink=0.5)

#         vmin = np.min([np.min(packsample[:,1,crop[0]:crop[1],crop[2]:crop[3]])])
#         vmax = np.min([np.max(packsample[:,1,crop[0]:crop[1],crop[2]:crop[3]])])
#         ax = fig.add_subplot(232)
#         ax.set_title("v real")
#         im = ax.imshow(packsample[mem_idx,1,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
#         fig.colorbar(im, shrink=0.5)
#         ax = fig.add_subplot(235)
#         im = ax.imshow(invsample[mem_idx,1,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
#         ax.set_title("v inv")
#         fig.colorbar(im, shrink=0.5)

#         vmin = np.min([np.min(packsample[:,2,crop[0]:crop[1],crop[2]:crop[3]])])
#         vmax = np.min([np.max(packsample[:,2,crop[0]:crop[1],crop[2]:crop[3]])])
#         ax = fig.add_subplot(233)
#         ax.set_title("t2m real")
#         im = ax.imshow(packsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
#         fig.colorbar(im, shrink=0.5)
#         ax = fig.add_subplot(236)
#         im = ax.imshow(invsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
#         ax.set_title("t2m inv")
#         fig.colorbar(im, shrink=0.5)

#         fig.suptitle(figtitle)
#         fig.tight_layout()
#         try:
#             fig.savefig(figname, dpi=100)
#         except Exception:
#             print(f"unable to save figure: {figname}")
#         plt.close()
#         return


# def online_inv_plot_2(packsample, invsample, crop=[0,-1,0,-1], mem_idx=0, figtitle=" ", figname="inv.png"):

#         fig = plt.figure(figsize=(20,20))

#         #### rr
#         vmin = np.min([np.min(packsample[:,0,crop[0]:crop[1],crop[2]:crop[3]])])
#         vmax = np.min([np.max(packsample[:,0,crop[0]:crop[1],crop[2]:crop[3]])])
#         ax = fig.add_subplot(341)
#         ax.set_title("rr real")
#         im = ax.imshow(packsample[mem_idx,0,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
#         fig.colorbar(im, shrink=0.5)

#         ax = fig.add_subplot(345)
#         im = ax.imshow(invsample[mem_idx,0,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
#         ax.set_title("rr inv")
#         fig.colorbar(im, shrink=0.5)

#         ax = fig.add_subplot(349)
#         diff = packsample[mem_idx,0,crop[0]:crop[1],crop[2]:crop[3]] - invsample[mem_idx,0,crop[0]:crop[1],crop[2]:crop[3]]
#         im = ax.imshow(diff, origin="lower", cmap="RdYlGn")
#         im.set_clim(-0.1,0.1)
#         ax.set_title("diff")
#         fig.colorbar(im, shrink=0.5)


#         #### u
#         vmin = np.min([np.min(packsample[:,1,crop[0]:crop[1],crop[2]:crop[3]])])
#         vmax = np.min([np.max(packsample[:,1,crop[0]:crop[1],crop[2]:crop[3]])])

#         ax = fig.add_subplot(342)
#         ax.set_title("u real")
#         im = ax.imshow(packsample[mem_idx,1,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
#         fig.colorbar(im, shrink=0.5)

#         ax = fig.add_subplot(345)
#         im = ax.imshow(invsample[mem_idx,1,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
#         ax.set_title("u inv")
#         fig.colorbar(im, shrink=0.5)

#         ax = fig.add_subplot(3,4,10)
#         diff = packsample[mem_idx,1,crop[0]:crop[1],crop[2]:crop[3]] - invsample[mem_idx,1,crop[0]:crop[1],crop[2]:crop[3]]
#         im = ax.imshow(diff, origin="lower", cmap="RdYlGn")
#         im.set_clim(-0.1,0.1)
#         ax.set_title("diff")
#         fig.colorbar(im, shrink=0.5)


#         #### v
#         vmin = np.min([np.min(packsample[:,2,crop[0]:crop[1],crop[2]:crop[3]])])
#         vmax = np.min([np.max(packsample[:,2,crop[0]:crop[1],crop[2]:crop[3]])])

#         ax = fig.add_subplot(343)
#         ax.set_title("v real")
#         im = ax.imshow(packsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
#         fig.colorbar(im, shrink=0.5)

#         ax = fig.add_subplot(347)
#         im = ax.imshow(invsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
#         ax.set_title("v inv")
#         fig.colorbar(im, shrink=0.5)

#         ax = fig.add_subplot(3,4,11)
#         diff = packsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]] - invsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]]
#         im = ax.imshow(diff, origin="lower", cmap="RdYlGn")
#         im.set_clim(-0.1,0.1)
#         ax.set_title("diff")
#         fig.colorbar(im, shrink=0.5)

#        #### t2m
#         vmin = np.min([np.min(packsample[:,3,crop[0]:crop[1],crop[2]:crop[3]])])
#         vmax = np.min([np.max(packsample[:,3,crop[0]:crop[1],crop[2]:crop[3]])])

#         ax = fig.add_subplot(344)
#         ax.set_title("t2m real")
#         im = ax.imshow(packsample[mem_idx,3,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
#         fig.colorbar(im, shrink=0.5)

#         ax = fig.add_subplot(348)
#         im = ax.imshow(invsample[mem_idx,3,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
#         ax.set_title("t2m inv")
#         fig.colorbar(im, shrink=0.5)

#         ax = fig.add_subplot(3,4,12)
#         diff = packsample[mem_idx,3,crop[0]:crop[1],crop[2]:crop[3]] - invsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]]
#         im = ax.imshow(diff, origin="lower", cmap="RdYlGn")
#         im.set_clim(-0.1,0.1)
#         ax.set_title("diff")
#         fig.colorbar(im, shrink=0.5)
#         fig.suptitle(figtitle)
#         fig.tight_layout()
        
#         try:
#             fig.savefig(figname, dpi=100)
#         except Exception:
#             print(f"unable to save figure: {figname}")
#         plt.close()
#         return


# def online_pert_plot(packsample, invsample, pert_sample, crop=[0,-1,0,-1], mem_idx=0, figtitle=" ", figname="inv.png"):

#         fig = plt.figure(figsize=(15,15))
#         mem_pert_idx = np.random.randint(0, len(pert_sample)-1)
#         #### u
#         vmin = np.min([np.min(packsample[:,0,crop[0]:crop[1],crop[2]:crop[3]])])
#         vmax = np.min([np.max(packsample[:,0,crop[0]:crop[1],crop[2]:crop[3]])])

#         ax = fig.add_subplot(331)
#         ax.set_title("u real")
#         im = ax.imshow(packsample[mem_idx,0,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
#         fig.colorbar(im, shrink=0.5)

#         ax = fig.add_subplot(334)
#         im = ax.imshow(invsample[mem_idx,0,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
#         ax.set_title("u inv")
#         fig.colorbar(im, shrink=0.5)

#         ax = fig.add_subplot(337)
#         ax.set_title("u perturbated")
#         im = ax.imshow(pert_sample[mem_pert_idx,0,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
#         fig.colorbar(im, shrink=0.5)


#         #### v
#         vmin = np.min([np.min(packsample[:,1,crop[0]:crop[1],crop[2]:crop[3]])])
#         vmax = np.min([np.max(packsample[:,1,crop[0]:crop[1],crop[2]:crop[3]])])

#         ax = fig.add_subplot(332)
#         ax.set_title("v real")
#         im = ax.imshow(packsample[mem_idx,1,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
#         fig.colorbar(im, shrink=0.5)

#         ax = fig.add_subplot(335)
#         im = ax.imshow(invsample[mem_idx,1,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
#         ax.set_title("v inv")
#         fig.colorbar(im, shrink=0.5)

#         ax = fig.add_subplot(338)
#         ax.set_title("v perturbated")
#         im = ax.imshow(pert_sample[mem_pert_idx,1,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower")
#         fig.colorbar(im, shrink=0.5)


#         #### t2m
#         vmin = np.min([np.min(packsample[:,2,crop[0]:crop[1],crop[2]:crop[3]])])
#         vmax = np.min([np.max(packsample[:,2,crop[0]:crop[1],crop[2]:crop[3]])])

#         ax = fig.add_subplot(333)
#         im = ax.imshow(packsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
#         ax.set_title("t2m real")
#         fig.colorbar(im, shrink=0.5)

#         ax = fig.add_subplot(336)
#         im = ax.imshow(invsample[mem_idx,2,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
#         ax.set_title("t2m inv")-
#         fig.colorbar(im, shrink=0.5)

#         ax = fig.add_subplot(339)
#         ax.set_title("t2m perturbated")
#         im = ax.imshow(pert_sample[mem_pert_idx,2,crop[0]:crop[1],crop[2]:crop[3]], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
#         fig.colorbar(im, shrink=0.5)

#         fig.suptitle(figtitle)
#         fig.tight_layout()
#         try:
#             fig.savefig(figname, dpi=100)
#         except Exception:
#             print(f"unable to save figure: {figname}")
#         plt.close()
#         return



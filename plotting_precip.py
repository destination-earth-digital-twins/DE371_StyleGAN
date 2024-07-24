import os
import numpy as np


for index,name in enumerate(os.listdir('./results_EP/AMSE/inversion/')):
    filename = name 
    if filename.endswith(".npy_.npy"):
        # Trouver la position de "Rsemble_" dans le nom de fichier
        start_index = filename.find("Rsemble_")

        print(filename)
        end_index = filename.find(".npy", start_index) + len(".npy")

        if start_index != -1 and end_index != -1:
            extracted_part = filename[start_index:end_index]
            print("La partie extraite est:", extracted_part)

            data2plot_origin = np.load(f'./results_EP/AMSE/pack/{extracted_part}').astype(np.float32)[:, np.newaxis, :, :]

            print("Les données chargées ont la forme:", data2plot_origin.shape)
        else:
            print("La partie désirée n'a pas été trouvée dans le nom de fichier.")
    else:
        print(filename)
            # data2plot = np.load(f'./results_EP/AMSE/inversion/{filename}').astype(np.float32)[:,np.newaxis,:,:]
            # for j in range(16):
            #     fig, axes = plt.subplots(2, 4, figsize=(20, 10))

            #     for i in range(4):
            #         ax = axes[0, i]
            #         if i==0:

            #             im = ax.imshow(np.exp(data2plot_origin[j][0][i])-1,cmap=cmapRR, origin="lower")
            #             vmin = np.min(np.exp(data2plot_origin[j][0][i])-1)
            #             vmax = np.max(np.exp(data2plot_origin[j][0][i])-1)
            #             print(vmin)
            #             ax.set_title(f'Original Image {j+1}')
            #             ax.axis('off')
            #             cbar = fig.colorbar(im, ax=ax)
            #             cbar.ax.tick_params(labelsize=8)
            #         elif i==3:
            #             im = ax.imshow(np.exp(data2plot_origin[j][0][i])-1, cmap='coolwarm',origin='lower')
            #             ax.set_title(f'Original Image {j+1}')
            #             ax.axis('off')
            #             cbar = fig.colorbar(im, ax=ax)
            #             cbar.ax.tick_params(labelsize=8)
            #         else:
            #             im = ax.imshow(data2plot_origin[j][0][i], cmap='viridis',origin='lower')
            #             ax.set_title(f'Original Image {j+1}')
            #             ax.axis('off')
            #             cbar = fig.colorbar(im, ax=ax)
            #             cbar.ax.tick_params(labelsize=8)
            #         ax = axes[1, i]
            #         if i ==0:
            #             im = ax.imshow(np.exp(data2plot[j][0][i])-1,cmap=cmapRR,origin='lower',vmax =vmax,vmin=vmin)
            #             print(vmin)
            #             ax.set_title(f'Inv precip {j+1}')
            #             ax.axis('off')
            #             cbar = fig.colorbar(im, ax=ax)
            #             cbar.ax.tick_params(labelsize=8)
            #         elif i ==3:

            #             im = ax.imshow(data2plot[j][0][i], cmap='coolwarm', origin='lower')
            #             ax.set_title(f'Inv_t2m_{j+1}')
            #             ax.axis('off')
            #             cbar = fig.colorbar(im, ax=ax)
            #             cbar.ax.tick_params(labelsize=8)
                        
            #         else:
            #             im = ax.imshow(data2plot[j][0][i], cmap='viridis',origin='lower')
            #             ax.set_title(f'Inv {j+1}')
            #             ax.axis('off')
            #             cbar = fig.colorbar(im, ax=ax)
            #             cbar.ax.tick_params(labelsize=8)
                        
            #     fig.savefig(f'./samples_precip/EP_weights_tests/fig_membre_{j}_mse.pdf', dpi=100)

            # #plt.tight_layout()
            # #plt.show()
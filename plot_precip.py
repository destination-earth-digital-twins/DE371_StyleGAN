import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import os 

cmapRR = colors.ListedColormap(["white","mediumpurple","blue","dodgerblue","darkseagreen","seagreen","greenyellow","yellow", "navajowhite","sandybrown","darkorange","red","darkred","black"], name='from_list', N=None)

folder = ('/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/1_batch_test_nuage')
list_dir = os.listdir(folder)
for n,dir in enumerate(list_dir):
    path_inv_files = os.path.join(folder,dir,'inversion')
    path_pack_files = os.path.join(folder,dir,'pack')
    for n_files,filename in enumerate(os.listdir(path_inv_files)):
        
        if filename.endswith('1500.npy') and filename.startswith('invert'):
            path_file_inv = os.path.join(path_inv_files,filename)
            # # Trouver la position de "invert" dans la chaîne
            start_pos = filename.find("invertFsemble_")
            end_pos = filename.find("_1500")

            result = 'Rsemble_'+filename[start_pos + len("invertFsemble_"):end_pos] +'.npy'
            path_file_pack = os.path.join(path_pack_files,result)
            # # Trouver la position de "1500" dans la chaîne
            data2plot_origin = np.load(path_file_pack).astype(np.float32)[:,np.newaxis,:,:]
            data2plot = np.load(path_file_inv).astype(np.float32)[:,np.newaxis,:,:]#samples_precip/EP_weights_tests/AMSE/inversion/invertFsemble_Rsemble_Rien signif_2.npy_.npy').astype(np.float32)[:,np.newaxis,:,:]
# 
            for j in range(16):
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))

                for i in range(4):
                    # Plot original image
                    ax = axes[0, i]
                    #print('LA', data2plot_origin[j][0][i].shape)

                    if i==0:
            #            im = ax.imshow(np.exp((data2plot_origin[1][0][0]+1)*5.78319931/2)-1,cmap=cmapRR, origin="lower")
            #           vmin = np.min(np.exp((data2plot_origin[1][0][0]+1)*5.78319931/2)-1)
            #          vmax = np.max(np.exp((data2plot_origin[1][0][0]+1)*5.78319931/2)-1)
                        im = ax.imshow(np.exp((data2plot_origin[j][0][i]+1)*5.78319931/2)-1,cmap=cmapRR, origin="lower")
                        vmin_rain = np.min(np.exp((data2plot_origin[j][0][i]+1)*5.78319931/2)-1)
                        vmax_rain = np.max(np.exp((data2plot_origin[j][0][i]+1)*5.78319931/2)-1)
                        print('LA',vmin_rain,vmax_rain)
                        ax.set_title(f'Original Image {j+1}')
                        ax.axis('off')
                        cbar = fig.colorbar(im, ax=ax)
                        cbar.ax.tick_params(labelsize=8)
                    elif i==3:
                    #  vmin = np.min([np.min(np.exp(data2plot_origin[j][0][i])-1)])
                    # vmax = np.min([np.max(np.exp(data2plot_origin[j][0][i])-1)])
                        im = ax.imshow((data2plot_origin[j][0][i]+1)*5.78319931/2, cmap='coolwarm',origin='lower')
                        ax.set_title(f'Original Image {j+1}')
                        ax.axis('off')
                        cbar = fig.colorbar(im, ax=ax)
                        cbar.ax.tick_params(labelsize=8)
                        vmin_temp = np.min((data2plot_origin[j][0][i]+1)*5.78319931/2)
                        vmax_temp = np.max((data2plot_origin[j][0][i]+1)*5.78319931/2)
                    elif i==2:
                        im = ax.imshow((data2plot_origin[j][0][i]+1)*5.78319931/2, cmap='viridis',origin='lower')
                        ax.set_title(f'Original Image {j+1}')
                        ax.axis('off')
                        cbar = fig.colorbar(im, ax=ax)
                        cbar.ax.tick_params(labelsize=8)
                        vmin_v = np.min((data2plot_origin[j][0][i]+1)*5.78319931/2)
                        vmax_v= np.max((data2plot_origin[j][0][i]+1)*5.78319931/2)
                    
                    elif i==1:
                        im = ax.imshow((data2plot_origin[j][0][i]+1)*5.78319931/2, cmap='viridis',origin='lower')
                        ax.set_title(f'Original Image {j+1}')
                        ax.axis('off')
                        cbar = fig.colorbar(im, ax=ax)
                        cbar.ax.tick_params(labelsize=8)
                        vmin_u = np.min((data2plot_origin[j][0][i]+1)*5.78319931/2)
                        vmax_u= np.max((data2plot_origin[j][0][i]+1)*5.78319931/2)

                    # Add noise and plot noisy image
                    ax = axes[1, i]
                    if i ==0:
                        im = ax.imshow(np.exp((data2plot[j][0][i]+1)*5.78319931/2)-1,cmap=cmapRR,origin='lower',vmax =vmax_rain,vmin=vmin_rain)
                        ax.set_title(f'Inv precip {j+1}')
                        ax.axis('off')
                        cbar = fig.colorbar(im, ax=ax)
                        cbar.ax.tick_params(labelsize=8)
                    elif i ==3:

                        im = ax.imshow((data2plot[j][0][i]+1)*5.78319931/2, cmap='coolwarm', origin='lower',vmin=vmin_temp,vmax=vmax_temp)
                        ax.set_title(f'Inv_t2m_{j+1}')
                        ax.axis('off')
                        cbar = fig.colorbar(im, ax=ax)
                        cbar.ax.tick_params(labelsize=8)
                    elif i==1:
                        im = ax.imshow((data2plot[j][0][i]+1)*5.78319931/2, cmap='viridis',origin='lower',vmin=vmin_u,vmax=vmax_u)
                        ax.set_title(f'Inv {j+1}')
                        ax.axis('off')
                        
                        cbar = fig.colorbar(im, ax=ax)
                        cbar.ax.tick_params(labelsize=8)
                        
                    elif i==2:
                        im = ax.imshow((data2plot[j][0][i]+1)*5.78319931/2, cmap='viridis',origin='lower',vmin=vmin_v,vmax=vmax_v)
                        ax.set_title(f'Inv {j+1}')
                        ax.axis('off')
                        
                        cbar = fig.colorbar(im, ax=ax)
                        cbar.ax.tick_params(labelsize=8)
                        


            plot_fold= os.path.join(folder,dir,'plot')
            print(plot_fold)
            if not os.path.exists(plot_fold):
                os.makedirs(plot_fold)
            if not os.path.exists(plot_fold):
                os.makedirs(plot_fold)
            plot_fold = os.path.join(folder,dir,'plot',filename[start_pos + len("invertFsemble_"):end_pos])
            fig.savefig(f'{plot_fold}_membre_{j+1}_1500.png', dpi=100)

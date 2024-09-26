import os
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Importer make_axes_locatable pour placer la colorbar
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
# Ce fichier permet de plotter sur une meme figure les diff loss pour 1 scenario


# Chemin du dossier principal
path_folder = '/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/BONS/VGG/tr/sol5_noise'
sub_dirs = ['mse', 'mae', 'wamse']#, 'wamse','amse','mul_pixel_loss_mae','mul_pixel_loss_mse']
sub_dir_new_path = 'inversion/Alpes-Mar_Golfe-G/'  # Remplace par le chemin correct
Rsemble = '/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/BON_moitie/mse/pack/Alpes-Mar_Golfe-G/Rsemble_Alpes-Mar_Golfe-G_0_batch2.npy'
# Initialiser une liste pour stocker les figures
figures = []

cmapRR = colors.ListedColormap(["white","mediumpurple","blue","dodgerblue","darkseagreen","seagreen","greenyellow","yellow", "navajowhite","sandybrown","darkorange","red","darkred","black"], name='from_list', N=None)

# Parcourir chaque sous-dossier
    # Construire le chemin vers le sous-dossier
    # Trouver les fichiers .npy dans le sous-dossier
    
    # Créer une figure pour chaque sous-dossier
fig, axs = plt.subplots(4, len(sub_dirs) + 1, figsize=(20, 12))

data2plot_origin = np.load(Rsemble)
titles = ['rr', 'u', 'v', 't2m']

for j in range(4):
    if j == 0:
        im = axs[j, 0].imshow(np.exp((data2plot_origin[0][j] + 1) * 5.78319931 / 2) - 1, cmap=cmapRR, origin="lower")
        axs[j, 0].axis('off')
        axs[j, 0].set_title(f'AROME - Alpes-Mar_Golfe-G')
        # vmin_rain = np.min((data2plot_origin[0][j] + 1) * 5.78319931 / 2 - 1)
        # vmax_rain = np.max((data2plot_origin[0][j] + 1) * 5.78319931 / 2 - 1)
        vmin_rain = np.min(np.exp((data2plot_origin[0][j]+1)*5.78319931/2)-1)
        vmax_rain = np.max(np.exp((data2plot_origin[0][j]+1)*5.78319931/2)-1)
        # Ajout d'une colorbar pour 'rr' (première ligne)
        divider = make_axes_locatable(axs[j, len(sub_dirs)])  # Utiliser la dernière colonne pour la colorbar
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        
    elif j == 1:
        im = axs[j, 0].imshow((data2plot_origin[0][j] + 1) * 5.78319931 / 2, cmap='viridis', origin='lower')
        axs[j, 0].axis('off')
        axs[j, 0].set_title(f'AROME - Alpes-Mar_Golfe-G')
        vmin_u = np.min((data2plot_origin[0][j]+1)*5.78319931/2)
        vmax_u= np.max((data2plot_origin[0][j]+1)*5.78319931/2)
        # Ajout d'une colorbar pour 'u' (deuxième ligne)
        divider = make_axes_locatable(axs[j, len(sub_dirs)])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        
    elif j == 2:
        im = axs[j, 0].imshow((data2plot_origin[0][j] + 1) * 5.78319931 / 2, cmap='viridis', origin='lower')
        axs[j, 0].axis('off')
        axs[j, 0].set_title(f'AROME - Alpes-Mar_Golfe-G')
        vmin_v = np.min((data2plot_origin[0][j]+1)*5.78319931/2)
        vmax_v= np.max((data2plot_origin[0][j]+1)*5.78319931/2)
        
        # Ajout d'une colorbar pour 'v' (troisième ligne)
        divider = make_axes_locatable(axs[j, len(sub_dirs)])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        
    elif j == 3:
        im = axs[j, 0].imshow((data2plot_origin[0][j] + 1) * 5.78319931 / 2, cmap='coolwarm', origin='lower')
        axs[j, 0].axis('off')
        axs[j, 0].set_title(f'AROME - Alpes-Mar_Golfe-G')
        vmin_t2m = np.min((data2plot_origin[0][j]+1)*5.78319931/2)
        vmax_t2m= np.max((data2plot_origin[0][j]+1)*5.78319931/2)
        
        # Ajout d'une colorbar pour 't2m' (quatrième ligne)
        divider = make_axes_locatable(axs[j, len(sub_dirs)])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

# Ajouter les sous-dossiers dans les colonnes
for i, sub_dir in enumerate(sub_dirs):
    current_sub_dir_path = os.path.join(path_folder, sub_dir)
    data = np.load(os.path.join(current_sub_dir_path, sub_dir_new_path, 'invertFsemble_Alpes-Mar_Golfe-G_0_batch2_1500_.npy'))
    
    for j in range(4):
        if j == 0:
            print('je suis la')
            im = axs[j, i + 1].imshow(np.exp((data[0][j]+1)*5.78319931/2)-1, cmap=cmapRR, origin='lower', vmax=vmax_rain, vmin=vmin_rain)
            axs[j, i + 1].axis('off')
            axs[j, i + 1].set_title(f'{sub_dir} - Alpes-Mar_Golfe-G')
            print('LA1',vmax_rain,vmin_rain,np.max(np.exp((data[0][j]+1)*5.78319931/2)-1),np.min(np.exp((data[0][j]+1)*5.78319931/2)-1))
        elif j == 1:
            data = np.where(np.isfinite(data),data,0)
            im = axs[j, i + 1].imshow((data[0][j]+1)*5.78319931/2, cmap='viridis', origin='lower',clim =(vmin_u,vmax_u))#vmin=vmin_u,vmax=vmax_u)
            axs[j, i + 1].axis('off')
            axs[j, i + 1].set_title(f'{sub_dir} - Alpes-Mar_Golfe-G')
            print('VMINU',vmin_u,'VMAX',vmax_u,np.where(np.isfinite(data)))

            print('VMINU2',np.min((data[0][j]+1)*5.78319931/2),np.max((data[0][j]+1)*5.78319931/2))
        elif j == 2:
            im = axs[j, i + 1].imshow((data[0][j]+1)*5.78319931/2, cmap='viridis', origin='lower',vmin=vmin_v,vmax=vmax_v)
            axs[j, i + 1].axis('off')
            axs[j, i + 1].set_title(f'{sub_dir} - Alpes-Mar_Golfe-G')
        elif j == 3:
            im = axs[j, i + 1].imshow((data[0][j]+1)*5.78319931/2, cmap='coolwarm', origin='lower',vmin=vmin_t2m,vmax=vmax_t2m)
            axs[j, i + 1].axis('off')
            axs[j, i + 1].set_title(f'{sub_dir} - Alpes-Mar_Golfe-G')

# Ajouter les titres/légendes
for j, title in enumerate(titles):
    axs[j, 0].set_ylabel(title)

plt.tight_layout()
plt.savefig(f'Alpes-Mar_Golfe-G_loss_member_0_vgg_tr_sol5_noise.png')
figures.append(fig)



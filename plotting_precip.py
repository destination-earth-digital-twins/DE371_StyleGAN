import os
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Importer make_axes_locatable pour placer la colorbar
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from pathlib import Path

# Ce fichier permet de plotter sur une meme figure les diff loss pour 1 scenario

# Chemin du dossier principal
scenarios = ['Alpes-Mar_Golfe-G','Cevennes_Gard_Hérault-N','Corse O','Rien signif']#,'Espagne','Espagne_Roussillon','Centre Médit','Jura_Alpes_Drôme-N','Massif-C Sud','Massif-C Centre','Pyrénées','Var_PACA Ouest_Drôme-S']
for _,scenario in enumerate(scenarios):
    path_folder = '/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/VGG/rdm/sol2_noise/pixel_vgg'
    sub_dirs = ['mse','mae', 'amse']#,'VGG_seul']
    sub_dir_new_path = f'inversion/{scenario}'  # Remplace par le chemin correct
    Rsembles = f'/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/VGG/rdm/sol2_noise/VGG_seul/pack/{scenario}/'
    # Initialiser une liste pour stocker les figures
    for _,batch in enumerate(sorted(os.listdir(Rsembles))):
        print(batch.split('_'),batch.split('_')[-2:])
        figures = []

        cmapRR = colors.ListedColormap(["white","mediumpurple","blue","dodgerblue","darkseagreen","seagreen","greenyellow","yellow", "navajowhite","sandybrown","darkorange","red","darkred","black"], name='from_list', N=None)

        # Parcourir chaque sous-dossier
            # Construire le chemin vers le sous-dossier
            # Trouver les fichiers .npy dans le sous-dossier
            
            # Créer une figure pour chaque sous-dossier
        for member in range(16):
            fig, axs = plt.subplots(4,len(sub_dirs) + 1, figsize=(20, 12))
            # fig, axs = plt.subplots(4, len(sub_dirs) + 1, figsize=(20, 12))
            Rsemble = Rsembles + 'Rsemble_' + scenario + '_' + batch.split('_')[-2:][0] +'_' + batch.split('_')[-2:][1]
            data2plot_origin = np.load(Rsemble)
            titles = ['rr', 'u', 'v', 't2m']

            for j in range(4):
                if j == 0:
                    im = axs[j, 0].imshow(np.exp((data2plot_origin[member][j] + 1) * 5.78319931 / 2) - 1, cmap=cmapRR, origin="lower")
                    axs[j, 0].axis('off')
                    axs[j, 0].set_title(f'AROME - {scenario}')
                    # vmin_rain = np.min((data2plot_origin[0][j] + 1) * 5.78319931 / 2 - 1)
                    # vmax_rain = np.max((data2plot_origin[0][j] + 1) * 5.78319931 / 2 - 1)
                    vmin_rain = np.min(np.exp((data2plot_origin[member][j]+1)*5.78319931/2)-1)
                    vmax_rain = np.max(np.exp((data2plot_origin[member][j]+1)*5.78319931/2)-1)
                    # Ajout d'une colorbar pour 'rr' (première ligne)
                    divider = make_axes_locatable(axs[j, len(sub_dirs)])  # Utiliser la dernière colonne pour la colorbar
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(im, cax=cax)
                    
                elif j == 1:
                    im = axs[j, 0].imshow((data2plot_origin[member][j] + 1) * 5.78319931 / 2, cmap='viridis', origin='lower')
                    axs[j, 0].axis('off')
                    axs[j, 0].set_title(f'AROME - ')
                    vmin_u = np.min((data2plot_origin[member][j]+1)*5.78319931/2)
                    vmax_u= np.max((data2plot_origin[member][j]+1)*5.78319931/2)
                    # Ajout d'une colorbar pour 'u' (deuxième ligne)
                    divider = make_axes_locatable(axs[j, len(sub_dirs)])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(im, cax=cax)
                    
                elif j == 2:
                    im = axs[j, 0].imshow((data2plot_origin[member][j] + 1) * 5.78319931 / 2, cmap='viridis', origin='lower')
                    axs[j, 0].axis('off')
                    axs[j, 0].set_title(f'AROME - ')
                    vmin_v = np.min((data2plot_origin[member][j]+1)*5.78319931/2)
                    vmax_v= np.max((data2plot_origin[member][j]+1)*5.78319931/2)
                    
                    # Ajout d'une colorbar pour 'v' (troisième ligne)
                    divider = make_axes_locatable(axs[j, len(sub_dirs)])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(im, cax=cax)
                    
                elif j == 3:
                    im = axs[j, 0].imshow((data2plot_origin[member][j] + 1) * 5.78319931 / 2, cmap='coolwarm', origin='lower')
                    axs[j, 0].axis('off')
                    axs[j, 0].set_title(f'AROME - ')
                    vmin_t2m = np.min((data2plot_origin[member][j]+1)*5.78319931/2)
                    vmax_t2m= np.max((data2plot_origin[member][j]+1)*5.78319931/2)
                    
                    # Ajout d'une colorbar pour 't2m' (quatrième ligne)
                    divider = make_axes_locatable(axs[j, len(sub_dirs)])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(im, cax=cax)

            # Ajouter les sous-dossiers dans les colonnes
            for i, sub_dir in enumerate(sub_dirs):
                current_sub_dir_path = os.path.join(path_folder, sub_dir)
                data_path = f'invertFsemble_{scenario}_' + batch.split('_')[-2:][0] +'_' + batch.split('_')[-2:][1].split('.')[0] +'_1000_.npy'
                path = Path(os.path.join(current_sub_dir_path, sub_dir_new_path, data_path ))
                if path.exists():
                    data = np.load(os.path.join(current_sub_dir_path, sub_dir_new_path, data_path ))
                
                    for j in range(4):
                        if j == 0:
                            print('je suis la')
                            im = axs[j, i + 1].imshow(np.exp((data[member][j]+1)*5.78319931/2)-1, cmap=cmapRR, origin='lower', vmax=vmax_rain, vmin=vmin_rain)
                            axs[j, i + 1].axis('off')
                            axs[j, i + 1].set_title(f'{sub_dir} - {scenario}')
                            print('LA1',vmax_rain,vmin_rain,np.max(np.exp((data[0][j]+1)*5.78319931/2)-1),np.min(np.exp((data[0][j]+1)*5.78319931/2)-1))
                        elif j == 1:
                            data = np.where(np.isfinite(data),data,0)
                            im = axs[j, i + 1].imshow((data[member][j]+1)*5.78319931/2, cmap='viridis', origin='lower',clim =(vmin_u,vmax_u))#vmin=vmin_u,vmax=vmax_u)
                            axs[j, i + 1].axis('off')
                            axs[j, i + 1].set_title(f'{sub_dir} - { scenario}')
                            print('VMINU',vmin_u,'VMAX',vmax_u,np.where(np.isfinite(data)))

                            print('VMINU2',np.min((data[0][j]+1)*5.78319931/2),np.max((data[0][j]+1)*5.78319931/2))
                        elif j == 2:
                            im = axs[j, i + 1].imshow((data[member][j]+1)*5.78319931/2, cmap='viridis', origin='lower',vmin=vmin_v,vmax=vmax_v)
                            axs[j, i + 1].axis('off')
                            axs[j, i + 1].set_title(f'{sub_dir} - {scenario}')
                        elif j == 3:
                            im = axs[j, i + 1].imshow((data[member][j]+1)*5.78319931/2, cmap='coolwarm', origin='lower',vmin=vmin_t2m,vmax=vmax_t2m)
                            axs[j, i + 1].axis('off')
                            axs[j, i + 1].set_title(f'{sub_dir} - {scenario}')
                else:
                    # Le fichier n'existe pas, tu peux passer
                    print("Le fichier n'existe pas. Passage.")
                    
                # Ajouter les titres/légendes
                for j, title in enumerate(titles):
                    axs[j, 0].set_ylabel(title)

                plt.tight_layout()
                save_path = os.path.join(path_folder,'plots',f'{scenario}')
                file_name = '_sol_' + scenario +'_' + batch.split('_')[-2:][0] + '_' + batch.split('_')[-2:][1] + '_' 

                try:
                    os.makedirs(save_path, exist_ok=True)
                    print("Directory '%s' created successfully" )
                except OSError as error:
                    print("Directory '%s' can not be created")
                plt.savefig(f'{save_path}_{file_name}_{member}.png')
            #  plt.savefig(f'{save_path}_{file_name}_{member}.pdf')

                figures.append(fig)

#POUR FICHIERS:
# path_folder = '/home/users/u101957/DE371_StyleGAN/data_tests_precip'
# for i, file in enumerate(os.listdir(path_folder)):
#     current_sub_dir_path = os.path.join(path_folder, file)
#     print(current_sub_dir_path)
#     data = np.load(current_sub_dir_path).astype(np.float32)[:,np.newaxis,:,:]
#     sub_dir = os.path.basename(file).split('_')[0:2]
#     print('SIZE',data.shape,file)
#     for j in range(4):
#         if j == 0:
#             print('je suis la',data[0][0][j].shape)
#             im = axs[j, i + 1].imshow(np.exp((data[0][0][j]+1)*5.78319931/2)-1, cmap=cmapRR, origin='lower', vmax=vmax_rain, vmin=vmin_rain)
#             axs[j, i + 1].axis('off')
#             axs[j, i + 1].set_title(f'{sub_dir[0]}_{sub_dir[1]} ')
#         elif j == 1:
#             data = np.where(np.isfinite(data),data,0)
#             im = axs[j, i + 1].imshow((data[0][0][j]+1)*5.78319931/2, cmap='viridis', origin='lower',clim =(vmin_u,vmax_u))#vmin=vmin_u,vmax=vmax_u)
#             axs[j, i + 1].axis('off')
#             axs[j, i + 1].set_title(f'{sub_dir[0]}_{sub_dir[1]}')

#         elif j == 2:
#             im = axs[j, i + 1].imshow((data[0][0][j]+1)*5.78319931/2, cmap='viridis', origin='lower',vmin=vmin_v,vmax=vmax_v)
#             axs[j, i + 1].axis('off')
#             axs[j, i + 1].set_title(f'{sub_dir[0]}_{sub_dir[1]} ')
#         elif j == 3:
#             im = axs[j, i + 1].imshow((data[0][0][j]+1)*5.78319931/2, cmap='coolwarm', origin='lower',vmin=vmin_t2m,vmax=vmax_t2m)
#             axs[j, i + 1].axis('off')
#             axs[j, i + 1].set_title(f'{sub_dir[0]}_{sub_dir[1]} ')

# # Ajouter les titres/légendes
# for j, title in enumerate(titles):
#     axs[j, 0].set_ylabel(title)

# plt.tight_layout()
# plt.savefig(f'centre_medit_member_0_vgg_seul.png')
# figures.append(fig)





import os 
import numpy as np 
import matplotlib.pyplot as plt


def images_load(folder_path):
    list_imgs = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            print(filename)
            image=np.load(os.path.join(folder_path,filename))
            list_imgs.append(image)
    # print(len(list_imgs),list_imgs[0].shape)
    return list_imgs
    
def calc_propor(list_imgs,threshold):
    Total = []
    if len(threshold)==1:
        nber_imgs=[]
        for number in range(len(list_imgs)):
            members=[]
            for member in range(list_imgs[number].shape[0]):
                # print(member)
                # print(list_imgs[number][member][0].shape)
                image = list_imgs[number][member][0]
                pixel_above_thresh = np.sum(image>=threshold)
                porportion_per_member = pixel_above_thresh/(256*256)
                members.append(porportion_per_member)
            # print(len(members), members)
            nber_imgs.append(np.mean(members))
        # print(np.log(nber_imgs),len(nber_imgs))
        return(np.mean(nber_imgs))
    else:
        th = []

        for _,thresh in enumerate(threshold):
            
            nber_imgs=[]
            for number in range(len(list_imgs)):
                members=[]
                for member in range(list_imgs[number].shape[0]):
                    image = np.exp((list_imgs[number][member][0]+1)*5.78319931/2)-1
                    pixel_above_thresh = np.sum(image>=thresh)
                    porportion_per_member = pixel_above_thresh/(256*256)
                    members.append(porportion_per_member)
                #print(len(members), members)
                nber_imgs.append(np.mean(members))
           # print(np.log(nber_imgs),len(nber_imgs))
            th.append((thresh,np.mean(nber_imgs)))
            #print(len(th),thresh,th)
        Total.append(th)
#print('TRESH',thresh,np.mean(nber_imgs),len(nber_imgs))

        return(th)
        

dossier_pack = '/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/scenarios/mse/pack'
dossier_inv = '/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/scenarios/mse/inversion'
list_imgs = images_load('/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/scenarios/mse/pack')
print(list_imgs)
#   i   ci, on obtient pour un threshold et un scénario la moyenne du nombre de pixels <= a ce seuil 
# pour continuer il faut créer une liste vide et la remplir de cette valeur pour tous les scénarios puis tracer dans un premier temps 
# scenarios_pack=[]
# thresh= [0,1,5,10]
# for i,folder in enumerate(os.listdir(dossier_pack)):
#     path_to_folder = os.path.join(dossier_pack,folder)
#     list_imgs = images_load(path_to_folder)
#     scenarios_pack.append(calc_propor(list_imgs,thresh))
#     data = calc_propor(list_imgs,thresh)
#     thresh = [item[0] for item in data]
#     values = [item[1] for item in data]
#     # # Tracer les données
#     # plt.figure()
#     # plt.plot(thresh, values, marker='o')
#     # plt.xlabel('Thresh')
#     # plt.ylabel('Value')
#     # plt.yscale('log')
#     # plt.title('Graph of Values vs Thresh')
#     # plt.grid(True)
#     # print(path_to_folder)
    
#     # # Sauvegarder la figure
#     # plt.savefig(os.path.join(path_to_folder,'figure_thresh_pack.png'))
# scenarios_inv=[]

# for i,folder in enumerate(os.listdir(dossier_inv)):
#     path_to_folder = os.path.join(dossier_inv,folder)
#     list_imgs = images_load(path_to_folder)
#     scenarios_inv.append(calc_propor(list_imgs,thresh))
#     data = calc_propor(list_imgs,thresh)
#     thresh = [item[0] for item in data]
#     values = [item[1] for item in data]
    

# assert len(scenarios_inv) == len(scenarios_pack), "Les deux listes doivent avoir la même longueur"

# # Boucle sur chaque sous-liste
# for i in range(len(scenarios_inv)):
#     # Séparer les a et les values pour premiereliste[i]
#     a_thresh = [item[0] for item in scenarios_inv[i]]
#     values_premiere = [item[1] for item in scenarios_inv[i]]
    
#     # Séparer les a et les values pour deuxiemeliste[i]
#     a_thresh = [item[0] for item in scenarios_pack[i]]
#     values_deuxieme = [item[1] for item in scenarios_pack[i]]
    
#     # Tracer les données
#     plt.figure()
#     plt.plot(a_thresh, values_premiere, marker='o', label='Inversion_scenarios')
#     plt.plot(a_thresh, values_deuxieme, marker='x', label='Scenarios Arome ')
#     plt.xlabel('a')
#     plt.ylabel('Value')
#     plt.title(f'Graphique {i+1}')
#     plt.yscale('log')  # Mettre l'échelle en ordonnées en log10
#     plt.grid(True, which="both", ls="--")
#     plt.legend()
    
#     # Sauvegarder la figure
#     plt.savefig(f'figure_{i+1}.png')

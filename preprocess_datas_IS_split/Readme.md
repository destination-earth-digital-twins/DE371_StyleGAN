# Pre-processing, importance sampling and csv file

Add observation handling.
Avant d'entrainer un dataset contenant des précipitations, il est important d'utiliser l'échnatillonnage préférentiel (EP) --> importance sampling. 
Comme résumé ci-dessous, cela nous permet d'éviter une sur représentation des samples sans précipitations. 

## Importance samling (IS) : 

IS allows us to avoid over-representing samples without precipitations.

L'importance sampling qu'est ce que c'est :  L’idée est de hiérarchiser
les données utilisées en leur conférant une "importance" au regard de leur contribution, par exemple, à
la variance totale du jeu de données. Les données les plus importantes sont sélectionnées avec une plus
grande probabilité et utilisées a posteriori pour l’entraînement du réseau. En résumé, cela sert à supprimer les samples qui 
contiennt peu ou pas de précipitations pour forcer le réseau à apprendre à reproduire les précipitations. 

Pour se faire il faut faire un pré-processing du dataset, pour rassembler nos samples en Gigafile (batchs):
                        " pre_pro_for_is.py"
Ensuite il faut utiliser le fichier : " called/process_is.py " afin d'appliquer l'EP. 

Nous allons reproduire n fois ce processus, c'est ce qu'on appelle le bootstrap, pour obtenir plusieurs fichihers csv résultants de différents EP. 

Ensuite, nous allons assembler ces différents csv pour agrandir notre train set: " bootrstaps.py"

Enfin on divise le dataset en train/test/valid set. 

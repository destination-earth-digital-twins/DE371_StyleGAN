# Pre-processing, importance sampling and csv file

Add observation handling.
Avant d'entrainer un dataset contenant des précipitations, il est important d'utiliser l'échnatillonnage préférentiel (EP) --> importance sampling. 
Comme résumé ci-dessous, cela nous permet d'éviter une sur représentation des samples sans précipitations. 

## Importance samling (IS) : 

IS allows us to avoid over-representing samples without precipitations.
We do this by giving an "importance" based on their contribution --> the most important datas are selected with a higher probability and then, used for the network training.
To resume, it removes  samples with little or no precipitation to encourage the network to focus on learning how to reproduce precipitation patterns.



Pour se faire il faut faire un pré-processing du dataset, pour rassembler nos samples en Gigafile (batchs):
                        " pre_pro_for_is.py"
Ensuite il faut utiliser le fichier : " called/process_is.py " afin d'appliquer l'EP. 

Nous allons reproduire n fois ce processus, c'est ce qu'on appelle le bootstrap, pour obtenir plusieurs fichihers csv résultants de différents EP. 

Ensuite, nous allons assembler ces différents csv pour agrandir notre train set: " bootrstaps.py"

Enfin on divise le dataset en train/test/valid set. 

# Pre-processing, importance sampling and csv file

Add observation handling.
Avant d'entrainer un dataset contenant des précipitations, il est important d'utiliser l'échnatillonnage préférentiel (EP) --> importance sampling. 
Comme résumé ci-dessous, cela nous permet d'éviter une sur représentation des samples sans précipitations. 

## Importance samling (IS) : 

IS allows us to avoid over-representing samples without precipitations.
We do this by giving an "importance" based on their contribution --> the most important datas are selected with a higher probability and then, used for the network training.
To resume, it removes  samples with little or no precipitation to encourage the network to focus on learning how to reproduce precipitation patterns.




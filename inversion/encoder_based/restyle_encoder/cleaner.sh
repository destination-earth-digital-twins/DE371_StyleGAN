#!/bin/bash
# https://github.com/NVIDIA/nvidia-docker/issues/1026


HOME_DIR="/home/mrmn/brochetc/restyle-encoder/" 
DATA_DIR="/scratch/mrmn/brochetc/GAN_2D/"


############################
HOROVOD_CONTAINER="stylegan4arome_1" #container name

#your user and group IDs -> needed to give you permissions for your working directories 
UID="$((`id -u`))"
GID="$((`id -g`))"

#DOCKER_MOUNTS_OPTS=" -v $(echo ~/.ssh):/tmp/.ssh:ro -v /scratch:/scratch -v /home:/home"  #mount the different directories 
DOCKER_MOUNTS_OPTS=" -v ${HOME_DIR}:${HOME_DIR} -v ${DATA_DIR}:${DATA_DIR}"  #mount the different directories


srun --job-name=brochet-interactive --partition=node1 --gres=gpu:v100:1 --time=10:00:00 --ntasks-per-node=1 docker run --rm ${DOCKER_OPTS} ${DOCKER_MOUNTS_OPTS} stylegan4arome_1 chmod +x ${HOME_DIR}slurm-docker-run_perso

exit
exit

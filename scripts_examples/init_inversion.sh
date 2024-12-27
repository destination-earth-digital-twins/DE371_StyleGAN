#!/bin/bash -l
#SBATCH -J DE371_stylegan
#SBATCH -A p200177
#SBATCH -q default
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00

export OMP_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda-12.1
export NVHPC_CUDA_HOME=/usr/local/cuda-12.1
export CXX=g++ #the compiler for cpp extensions
export CC=gcc  #the compiler to access the good cpp standard
export APPTAINER_BINDPATH="/PATH/TO/datasets/,/PATH/TO/DE_371:/PATH/TO/DE_371/"
module load Apptainer/1.2.4-GCCcore-12.3.0


# Inversion with pixel loss 

apptainer exec --nv /PATH/TO/apptainer_container/container.sif python3 main_inversion.py \
        --invstep=2000\
        --pixel_loss="amse"\
        --lambda_vgg=0\
        --device='cuda:0'\
        --output_dir='/PATH/TO/inversion/'\
        --pack_dir='/PATH/TO/pack/'\

# Inversion with perceptual loss 
# apptainer exec --nv /PATH/TO/apptainer_container/container.sif python3 main_inversion_precip.py \
#         --invstep=2000\
#         --lambda_pixel=0\
#         --lambda_lpips=0\
#         --vgg_computation='sol2'\
#         --device='cuda:1'\




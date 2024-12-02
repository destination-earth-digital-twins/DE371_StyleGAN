#!/bin/bash -l
#SBATCH -J DE371_stylegan
#SBATCH -A p200177
#SBATCH -q short
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-1:0:0

export OMP_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda-12.1
export NVHPC_CUDA_HOME=/usr/local/cuda-12.1
export CXX=g++ #the compiler for cpp extensions
export CC=gcc  #the compiler to access the good cpp standard
export APPTAINER_BINDPATH="/PATH/TO/datasets:/PATH/TO/datasets/,/PATH/TO/INV DATA:/PATH/TO/INV DATA/"
module load Apptainer/1.2.4-GCCcore-12.3.0

apptainer exec --nv /PATH/TO/apptainer_container/container.sif python3 /PATH/TO/scale_tune.py \
 
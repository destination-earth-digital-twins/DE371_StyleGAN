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
export APPTAINER_BINDPATH="/project/home/p200177/DE_371/:/project/home/p200177/DE_371/"
module load env/release/2023.1
module load env/staging/2023.1
module load Apptainer/1.2.4-GCCcore-12.3.0

apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container.sif torchrun --nproc_per_node=4 main_gan.py \
        --data_dir='/project/home/p200177/DE_371/datasets/dataset_MetNorway/MEPS_samples/' \
        --config_dir='/project/home/p200177/DE_371/datasets/dataset_MetNorway/MEPS_samples/' \
        --id_file='train_labels.csv' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP1/MEPS/training/2025-03-10/' \
        --g_channels=3 \
        --d_channels=3 \
        --epochs_num=30 \
        --var_names=['u','v','t2m'] \
        --use_noise='True' \


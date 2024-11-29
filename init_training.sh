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
export APPTAINER_BINDPATH="/project/home/p200177/DE_371/datasets:/project/home/p200177/DE_371/datasets/,/project/scratch/p200177/DE_371:/project/scratch/p200177/DE_371/"
module load Apptainer/1.2.4-GCCcore-12.3.0



apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif torchrun --nproc_per_node=4 main_gan.py \
        --data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France_rr_u_v_t2m/data/IS_rr_debug_1_1.0_0_0_0_0_0_256_large_lt/' \
        --id_file="IS_boostrap_no_duplicate_rr_cumul_correct_train.csv" \
        --output_dir='/project/scratch/p200177/DE_371/angeliquebonamy/training/gan_training_new_dataset/exp_train_ep_with_Noise_Injection/' \
        --g_channels=4 \
        --d_channels=4 \
        --epochs_num=30 \
        --var_names=['rr','u','v','t2m'] \
        --use_noise='True' \


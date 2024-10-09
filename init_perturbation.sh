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
export APPTAINER_BINDPATH="/project/home/p200177/DE_371:/project/home/p200177/DE_371/,/project/scratch/p200177/DE_371:/project/scratch/p200177/DE_371/"
module load Apptainer/1.2.4-GCCcore-12.3.0

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_perturbation.py \
        --ckpt_dir='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --data_dir='/project/home/p200177/DE_371/experiments_WP1/inversion_process_analysis/final_inversion_on_test_set/mse_only/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP1/inversion_process_analysis/final_inversion_on_test_set/pack_meanmax/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP1/inversion_process_analysis/final_inversion_on_test_set/mse_only/perturbation/' \
        --scale_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/scale_dir_gan_training/' \
        --eigendir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/eigenvalues_gan_training/' \
        --device='cuda' \
        --N_samples=50 \
        --N_conditioners=16 \
        --inv_step=1000 \
        --style_indices='[1,1,1,1,1,1,1,1,1,1,0,0,0,0]' \
        --date_start=2021-06-16 \
        --date_stop=2021-11-16 \
        --leadtimes='[3,6,9,12,15,18,21,24,27,30,33,36,39,42]' \
        --save_w_perturbated \
        
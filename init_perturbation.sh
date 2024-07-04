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
export APPTAINER_BINDPATH="/project/home/p200177/DE_371/datasets:/project/home/p200177/DE_371/datasets/,/project/scratch/p200177/DE_371/victorsanchez:/project/scratch/p200177/DE_371/victorsanchez/"
module load Apptainer/1.2.4-GCCcore-12.3.0

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_perturbation.py \
        --ckpt_dir='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --data_dir='/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise/Inversion_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise/' \
        --pack_dir='/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise/Pack_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise/' \
        --output_dir='/project/scratch/p200177/DE_371/victorsanchez/results/perturbation/Ens_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise/' \
        --scale_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/scale_dir_gan_training/' \
        --eigendir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/eigenvalues_gan_training/' \
        --device='cuda' \
        --N_samples=112 \
        --N_conditioners=16 \
        --style_indices='[1,1,1,1,1,1,1,1,1,1,0,0,0,0]' \
        --date_start=2021-07-01 \
        --date_stop=2021-07-31 \
        --leadtimes='[3,6,9,12,15,18,21,24,27,30,33,36,39,42]' \
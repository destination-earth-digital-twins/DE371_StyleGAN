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

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 scale_tune.py \
        --ckpt_dir='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/scratch/p200177/DE_371/victorsanchez/results/scaled_perturbation/ScaleTune/' \
        --ensemble_data_dir='/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise_no_noise_inj/Pack_Perceptual_Random_VGG_Loss_sol3_not_fixed_noise_no_noise_inj/' \
        --fake_data_dir='/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Ens_Perceptual_Random_VGG_Loss_sol3/Inversion_Perceptual_Random_VGG_Loss_sol3/' \
        
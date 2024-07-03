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

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Validation_Set_For_ScaleTune/Inversion/' \
        --pack_dir='/project/scratch/p200177/DE_371/victorsanchez/results/inversion/Validation_Set_For_ScaleTune/Pack_Perceptual/' \
        --device='cuda' \
        --lambda_pixel=10 \
        --lambda_vgg=1 \
        --vgg_computation='sol3' \
        --lambda_noise=0 \
        --noise_optimize=0 \
        --fixed_noise=0 \
        --invstep=2000 \
        --inv_checkpoints='[500,1000,2000]' \
        --dates_file='Large_lt_val_labels.csv' \
        --date_start=2020-06-15 \
        --date_stop=2021-06-14 \
        --leadtimes='[3,6,9,12,15,18,21,24,27,30,33,36,39,42]' \
        --progressive_loss_mode=1 \
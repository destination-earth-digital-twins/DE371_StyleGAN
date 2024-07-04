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
export APPTAINER_BINDPATH="/project/home/p200177/DE_371/datasets:/project/home/p200177/DE_371/datasets/,/project/scratch/p200177/DE_371/victorsanchez:/project/scratch/p200177/DE_371/victorsanchez/"
module load Apptainer/1.2.4-GCCcore-12.3.0

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        # --ckpt_dir='/home/users/u101957/DE371_StyleGAN/results/models/012000.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
       # --output_dir='/home/users/u101957/DE371_StyleGAN/results/Inversion_Perceptual_Random_VGG_Loss_sol3/' \
        --pack_dir='/home/users/u101957/DE371_StyleGAN/results/Pack_Perceptual_Random_VGG_Loss_sol3/' \
        --device='cuda' \
        --pixel_loss_type='amse'\ 
        --lambda_vgg=0 \
        --lambda_pixel=10 \
        --lambda_vgg=1 \
        --vgg_computation='sol3' \
        --lambda_noise=0 \
        --noise_optimize=0 \
        --invstep=2000 \
        --inv_checkpoints='[250,500,1000,1500,2000]' \
        --date_start=2021-07-01 \
        --date_stop=2021-07-31 \
        --leadtimes='[3,6,9,12,15,18,21,24,27,30,33,36,39,42]' \
        --progressive_loss_mode=1 \
        --normalization='meanmax'
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
export APPTAINER_BINDPATH="/project/home/p200177/DE_371/:/project/home/p200177/DE_371/,/project/scratch/p200177/DE_371:/project/scratch/p200177/DE_371/"
module load Apptainer/1.2.4-GCCcore-12.3.0

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 temporal_interpolation.py \
        --device='cuda' \
        --ckpt_dir='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt' \
        --latent_vectors_dir='/project/home/p200177/DE_371/experiments_WP1/inversion_process_analysis/inversion/exp34/inversion' \
        --output_dir='/project/scratch/p200177/DE_371/temporal_downscaling_experiments/u101834/2021-07-16' \
        --date=2021-07-16 \
        --input_leadtimes='[3,9,15,21,27,33,39]' \
        --ref_leadtimes='[6,12,18,24,30,33]' \
        --invstep=2000 \


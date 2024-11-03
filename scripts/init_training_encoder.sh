#!/bin/bash -l
#SBATCH -J DE371_stylegan
#SBATCH -A p200177
#SBATCH -q default
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00

export OMP_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda-12.1
export NVHPC_CUDA_HOME=/usr/local/cuda-12.1
export CXX=g++ #the compiler for cpp extensions
export CC=gcc  #the compiler to access the good cpp standard
export APPTAINER_BINDPATH="/project/home/p200177/DE_371:/project/home/p200177/DE_371/,/project/scratch/p200177/DE_371:/project/scratch/p200177/DE_371/"
module load Apptainer/1.2.4-GCCcore-12.3.0


apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container_encoder.sif python3 restyle_encoder/train_restyle_psp.py \
        --exp_dir='/project/scratch/p200177/DE_371/victorsanchez/results/encoder/' \
        --learning_rate=0.001 \
        --l2_lambda=0 \
        --vgg_lambda=1 \
        --w_norm_lambda=0 \
        --vgg_computation='sol2' \
        --max_steps=50000 \
        --start_from_latent_avg \
        --n_iters_per_batch=10 \
        --batch_size=8 \
        --test_batch_size=2 \

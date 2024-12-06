#!/bin/bash -l
#SBATCH -J DE371_stylegan
#SBATCH -A p200177
#SBATCH -q default
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda-12.1
export NVHPC_CUDA_HOME=/usr/local/cuda-12.1
export CXX=g++ #the compiler for cpp extensions
export CC=gcc  #the compiler to access the good cpp standard
export APPTAINER_BINDPATH="/project/home/p200177/DE_371/:/project/home/p200177/DE_371/,/project/scratch/p200177/DE_371:/project/scratch/p200177/DE_371/"
module load Apptainer/1.2.4-GCCcore-12.3.0

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 train_interpolator.py \
        --device='cuda:0' \
        --model_name='LatentInterpolatorCorrector' \
        --training_description='1024-3-perc1000' \
        --weight_decay=1e-5 \
        --learning_rate=1e-3 \
        --lr_decay=0.9 \
        --latent_loss_weight=0.0 \
        --pixel_loss_weight=0.0 \
        --perceptual_loss_weight=10.0 \
        --num_neurons=1024 \
        --normalization="Layer" \
        --dropout=0.0 \
        --num_layers=3 > training-9.log 2>&1 &

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 train_interpolator.py \
        --device='cuda:1' \
        --model_name='LatentInterpolatorCorrector' \
        --training_description='1024-3-perc50' \
        --weight_decay=1e-5 \
        --learning_rate=1e-3 \
        --lr_decay=0.9 \
        --latent_loss_weight=0.5 \
        --pixel_loss_weight=0.0 \
        --perceptual_loss_weight=0.5 \
        --num_neurons=1024 \
        --normalization="Layer" \
        --dropout=0.0 \
        --num_layers=3 > training-10.log 2>&1 &

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 train_interpolator.py \
        --device='cuda:2' \
        --model_name='LatentInterpolatorCorrector' \
        --training_description='1024-3-perc075' \
        --weight_decay=1e-5 \
        --learning_rate=1e-3 \
        --lr_decay=0.9 \
        --latent_loss_weight=0.25 \
        --pixel_loss_weight=0.0 \
        --perceptual_loss_weight=0.75 \
        --num_neurons=1024 \
        --normalization="Layer" \
        --dropout=0.0 \
        --num_layers=3 > training-11.log 2>&1 &

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 train_interpolator.py \
        --device='cuda:3' \
        --model_name='LatentInterpolatorCorrector' \
        --training_description='1024-3-perc100' \
        --weight_decay=1e-5 \
        --learning_rate=1e-3 \
        --lr_decay=0.9 \
        --latent_loss_weight=0.0 \
        --pixel_loss_weight=0.0 \
        --perceptual_loss_weight=1.0 \
        --num_neurons=1024 \
        --normalization="Layer" \
        --dropout=0.0 \
        --num_layers=3 > training-12.log 2>&1 &

wait

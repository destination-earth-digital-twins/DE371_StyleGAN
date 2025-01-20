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

export OMP_NUM_THREADS=4
export CUDA_HOME=/usr/local/cuda-12.1
export NVHPC_CUDA_HOME=/usr/local/cuda-12.1
export APPTAINERENV_CUDA_VISIBLE_DEVICES='0,1,2,3'
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export CXX=g++ #the compiler for cpp extensions
export CC=gcc  #the compiler to access the good cpp standard
export NCCL_ASYNC_ERROR_HANDLING=1
export APPTAINER_BINDPATH="/project/home/p200177/DE_371/:/project/home/p200177/DE_371/"
module load Apptainer/1.2.4-GCCcore-12.3.0

apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container.sif torchrun --nproc_per_node=4 --master_port=29500 train_interpolator.py \
        --model_name='StyleVectorInterpolatorCorrector' \
        --training_description='512-4-perc50-pixel-250-lat025' \
        --num_workers=16 \
        --weight_decay=0.0 \
        --learning_rate=1e-4 \
        --lr_decay=0.9 \
        --latent_loss_weight=0.25 \
        --pixel_loss_weight=250.0 \
        --perceptual_loss_weight=50.0 \
        --mae_loss \
        --num_neurons=512 \
        --normalization="Layer" \
        --dropout=0.0 \
        --epochs=20 \
        --batch_size=2 \
        --start_date=2020-06-15 \
        --end_date=2021-06-14 \
        --num_layers=4 > training-4.log 2>&1
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


apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_encoder_inversion.py \
        --checkpoint_path='/project/scratch/p200177/DE_371/victorsanchez/results/encoder/restyle_pSp_training/lr_0.001_vgg_lambda_1.0_resnet34=trained_10_iter/Instance_1/checkpoints/iteration_50000.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP1/encoder_inversion/exp1/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP1/encoder_inversion/exp1/pack/' \
        --device='cuda:0' \
        --date_start=2021-07-01 \
        --date_stop=2021-07-07 \
        --leadtimes='[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45]' \
        --plot_checkpoint  > enc_inv_0.log 2>&1 &

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_encoder_inversion.py \
        --checkpoint_path='/project/scratch/p200177/DE_371/victorsanchez/results/encoder/restyle_pSp_training/lr_0.001_vgg_lambda_1.0_resnet34=trained_10_iter/Instance_1/checkpoints/iteration_50000.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP1/encoder_inversion/exp1/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP1/encoder_inversion/exp1/pack/' \
        --device='cuda:1' \
        --date_start=2021-07-07 \
        --date_stop=2021-07-14 \
        --leadtimes='[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45]' \
        --plot_checkpoint  > enc_inv_1.log 2>&1 &

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_encoder_inversion.py \
        --checkpoint_path='/project/scratch/p200177/DE_371/victorsanchez/results/encoder/restyle_pSp_training/lr_0.001_vgg_lambda_1.0_resnet34=trained_10_iter/Instance_1/checkpoints/iteration_50000.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP1/encoder_inversion/exp1/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP1/encoder_inversion/exp1/pack/' \
        --device='cuda:2' \
        --date_start=2021-07-14 \
        --date_stop=2021-07-21 \
        --leadtimes='[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45]' \
        --plot_checkpoint  > enc_inv_2.log 2>&1 &

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_encoder_inversion.py \
        --checkpoint_path='/project/scratch/p200177/DE_371/victorsanchez/results/encoder/restyle_pSp_training/lr_0.001_vgg_lambda_1.0_resnet34=trained_10_iter/Instance_1/checkpoints/iteration_50000.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP1/encoder_inversion/exp1/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP1/encoder_inversion/exp1/pack/' \
        --device='cuda:3' \
        --date_start=2021-07-21 \
        --date_stop=2021-08-01 \
        --leadtimes='[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45]' \
        --plot_checkpoint  > enc_inv_3.log 2>&1 &

wait
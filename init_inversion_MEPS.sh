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
export APPTAINER_BINDPATH="/project/home/p200177/DE_371/:/project/home/p200177/DE_371/"
module load env/release/2023.1
module load env/staging/2023.1
module load Apptainer/1.2.4-GCCcore-12.3.0

leadtimes='[1,4,7,10,13,16]'

apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/home/p200177/DE_371/experiments_WP1/MEPS/training/2025-05-27/models/072000.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/datasets_SMHI/256x256/samples/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/pack/' \
        --device='cuda:0' \
        --g_channels=4 \
        --normalization='meanmax' \
        --lambda_pixel=0 \
        --lambda_perceptual_loss=1 \
        --lambda_ms_ssim=0 \
        --channel_computation='sol2' \
        --lambda_noise=1e5 \
        --noise_optimize \
        --invstep=1000 \
        --inv_checkpoints='[1000]' \
        --feature_layers='[0,1,2,3]' \
        --dates_file='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/labels.csv' \
        --mean_file='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/global_mean.npy' \
        --max_file='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/global_std.npy' \
        --date_start=2024-01-01 \
        --date_stop=2024-01-08 \
        --leadtimes="$leadtimes" \
        --network_type='vgg16' \
        --plot_checkpoint > inversion_MEPS_1.log 2>&1 &

apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/home/p200177/DE_371/experiments_WP1/MEPS/training/2025-05-27/models/072000.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/datasets_SMHI/256x256/samples/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/pack/' \
        --device='cuda:1' \
        --g_channels=4 \
        --normalization='meanmax' \
        --lambda_pixel=0 \
        --lambda_perceptual_loss=1 \
        --lambda_ms_ssim=0 \
        --channel_computation='sol2' \
        --lambda_noise=1e5 \
        --noise_optimize \
        --invstep=1000 \
        --inv_checkpoints='[1000]' \
        --feature_layers='[0,1,2,3]' \
        --dates_file='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/labels.csv' \
        --mean_file='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/global_mean.npy' \
        --max_file='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/global_std.npy' \
        --date_start=2024-01-08 \
        --date_stop=2024-01-15 \
        --leadtimes="$leadtimes" \
        --network_type='vgg16' \
        --plot_checkpoint > inversion_MEPS_2.log 2>&1 &

apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/home/p200177/DE_371/experiments_WP1/MEPS/training/2025-05-27/models/072000.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/datasets_SMHI/256x256/samples/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/pack/' \
        --device='cuda:2' \
        --g_channels=4 \
        --normalization='meanmax' \
        --lambda_pixel=0 \
        --lambda_perceptual_loss=1 \
        --lambda_ms_ssim=0 \
        --channel_computation='sol2' \
        --lambda_noise=1e5 \
        --noise_optimize \
        --invstep=1000 \
        --inv_checkpoints='[1000]' \
        --feature_layers='[0,1,2,3]' \
        --dates_file='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/labels.csv' \
        --mean_file='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/global_mean.npy' \
        --max_file='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/global_std.npy' \
        --date_start=2024-01-15 \
        --date_stop=2024-01-22 \
        --leadtimes="$leadtimes" \
        --network_type='vgg16' \
        --plot_checkpoint > inversion_MEPS_3.log 2>&1 &

apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/home/p200177/DE_371/experiments_WP1/MEPS/training/2025-05-27/models/072000.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/datasets_SMHI/256x256/samples/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/pack/' \
        --device='cuda:3' \
        --g_channels=4 \
        --normalization='meanmax' \
        --lambda_pixel=0 \
        --lambda_perceptual_loss=1 \
        --lambda_ms_ssim=0 \
        --channel_computation='sol2' \
        --lambda_noise=1e5 \
        --noise_optimize \
        --invstep=1000 \
        --inv_checkpoints='[1000]' \
        --feature_layers='[0,1,2,3]' \
        --dates_file='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/labels.csv' \
        --mean_file='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/global_mean.npy' \
        --max_file='/project/home/p200177/DE_371/experiments_WP1/MEPS/inversion/2025-05-28/global_std.npy' \
        --date_start=2024-01-22 \
        --date_stop=2024-01-29 \
        --leadtimes="$leadtimes" \
        --network_type='vgg16' \
        --plot_checkpoint > inversion_MEPS_4.log 2>&1 &
wait
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
module load Apptainer/1.2.4-GCCcore-12.3.0

leadtimes='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]'

apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/home/p200177/DE_371/resources/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_test/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_test/pack/' \
        --device='cuda:0' \
        --lambda_pixel=0 \
        --lambda_perceptual_loss=1 \
        --lambda_ms_ssim=0 \
        --channel_computation='sol2' \
        --lambda_noise=1e5 \
        --noise_optimize \
        --invstep=1000 \
        --inv_checkpoints='[1000]' \
        --feature_layers='[0,1,2,3]' \
        --dates_file='Large_lt_test_labels.csv' \
        --date_start=2021-10-15 \
        --date_stop=2021-10-23 \
        --leadtimes="$leadtimes" \
        --network_type='vgg16' \
        --plot_checkpoint > inversion_test_9.log 2>&1 &

apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/home/p200177/DE_371/resources/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_test/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_test/pack/' \
        --device='cuda:1' \
        --lambda_pixel=0 \
        --lambda_perceptual_loss=1 \
        --lambda_ms_ssim=0 \
        --channel_computation='sol2' \
        --lambda_noise=1e5 \
        --noise_optimize \
        --invstep=1000 \
        --inv_checkpoints='[1000]' \
        --feature_layers='[0,1,2,3]' \
        --dates_file='Large_lt_test_labels.csv' \
        --date_start=2021-10-23 \
        --date_stop=2021-11-01 \
        --leadtimes="$leadtimes" \
        --network_type='vgg16' \
        --plot_checkpoint > inversion_test_10.log 2>&1 &

apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/home/p200177/DE_371/resources/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_test/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_test/pack/' \
        --device='cuda:2' \
        --lambda_pixel=0 \
        --lambda_perceptual_loss=1 \
        --lambda_ms_ssim=0 \
        --channel_computation='sol2' \
        --lambda_noise=1e5 \
        --noise_optimize \
        --invstep=1000 \
        --inv_checkpoints='[1000]' \
        --feature_layers='[0,1,2,3]' \
        --dates_file='Large_lt_test_labels.csv' \
        --date_start=2021-11-01 \
        --date_stop=2021-11-08 \
        --leadtimes="$leadtimes" \
        --network_type='vgg16' \
        --plot_checkpoint > inversion_test_11.log 2>&1 &

apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/home/p200177/DE_371/resources/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_test/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_test/pack/' \
        --device='cuda:3' \
        --lambda_pixel=0 \
        --lambda_perceptual_loss=1 \
        --lambda_ms_ssim=0 \
        --channel_computation='sol2' \
        --lambda_noise=1e5 \
        --noise_optimize \
        --invstep=1000 \
        --inv_checkpoints='[1000]' \
        --feature_layers='[0,1,2,3]' \
        --dates_file='Large_lt_test_labels.csv' \
        --date_start=2021-11-08 \
        --date_stop=2021-11-15 \
        --leadtimes="$leadtimes" \
        --network_type='vgg16' \
        --plot_checkpoint > inversion_test_12.log 2>&1 &
wait
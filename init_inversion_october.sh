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

leadtimes='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]'

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_october/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_october/pack/' \
        --device='cuda:0' \
        --lambda_pixel=0 \
        --lambda_vgg=1 \
        --lambda_ms_ssim=0 \
        --vgg_computation='sol2' \
        --lambda_noise=1e5 \
        --noise_optimize \
        --invstep=1000 \
        --inv_checkpoints='[250,500,750,1000]' \
        --vgg_feature_layers='[0,1,2,3]' \
        --date_start=2021-10-29 \
        --date_stop=2021-10-30 \
        --leadtimes="$leadtimes" \
        --vgg_state_dict_path='/project/scratch/p200177/DE_371/resources/vgg_weights/vgg16-random.pth' \
        --plot_checkpoint > inversion_oct_1_short.log 2>&1 &

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_october/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_october/pack/' \
        --device='cuda:1' \
        --lambda_pixel=0 \
        --lambda_vgg=1 \
        --lambda_ms_ssim=0 \
        --vgg_computation='sol2' \
        --lambda_noise=1e5 \
        --noise_optimize \
        --invstep=1000 \
        --inv_checkpoints='[250,500,750,1000]' \
        --vgg_feature_layers='[0,1,2,3]' \
        --date_start=2021-10-30 \
        --date_stop=2021-10-31 \
        --leadtimes="$leadtimes" \
        --vgg_state_dict_path='/project/scratch/p200177/DE_371/resources/vgg_weights/vgg16-random.pth' \
        --plot_checkpoint > inversion_oct_2_short.log 2>&1 &

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_october/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_october/pack/' \
        --device='cuda:2' \
        --lambda_pixel=0 \
        --lambda_vgg=1 \
        --lambda_ms_ssim=0 \
        --vgg_computation='sol2' \
        --lambda_noise=1e5 \
        --noise_optimize \
        --invstep=1000 \
        --inv_checkpoints='[250,500,750,1000]' \
        --vgg_feature_layers='[0,1,2,3]' \
        --date_start=2021-10-31 \
        --date_stop=2021-11-01 \
        --leadtimes="$leadtimes" \
        --vgg_state_dict_path='/project/scratch/p200177/DE_371/resources/vgg_weights/vgg16-random.pth' \
        --plot_checkpoint > inversion_oct_3_short.log 2>&1 &

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_october/inversion/' \
        --pack_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/inversion_october/pack/' \
        --device='cuda:3' \
        --lambda_pixel=0 \
        --lambda_vgg=1 \
        --lambda_ms_ssim=0 \
        --vgg_computation='sol2' \
        --lambda_noise=1e5 \
        --noise_optimize \
        --invstep=1000 \
        --inv_checkpoints='[250,500,750,1000]' \
        --vgg_feature_layers='[0,1,2,3]' \
        --date_start=2021-11-01 \
        --date_stop=2021-11-02 \
        --leadtimes="$leadtimes" \
        --vgg_state_dict_path='/project/scratch/p200177/DE_371/resources/vgg_weights/vgg16-random.pth' \
        --plot_checkpoint > inversion_oct_4_short.log 2>&1 &

wait

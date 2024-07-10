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
export APPTAINER_BINDPATH="/project/home/p200177/DE_371/datasets:/project/home/p200177/DE_371/datasets/,/project/scratch/p200177/DE_371:/project/scratch/p200177/DE_371/"
module load Apptainer/1.2.4-GCCcore-12.3.0

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/scratch/p200177/DE_371/inversion_experiments/exp21/inversion/' \
        --pack_dir='/project/scratch/p200177/DE_371/inversion_experiments/exp21/pack/' \
        --device='cuda:0' \
        --lambda_pixel=0 \
        --lambda_vgg=1 \
        --vgg_computation='sol3' \
        --lambda_noise=0 \
        --invstep=1000 \
        --inv_checkpoints='[250,500,1000,1500,2000]' \
        --date_start=2021-07-01 \
        --date_stop=2021-07-17 \
        --leadtimes='[3,6,9,12,15,18,21,24,27,30,33,36,39,42]' \
        --vgg_state_dict_path='/project/scratch/p200177/DE_371/resources/vgg_weights/vgg16-random.pth' \
        --plot_checkpoint > exp21.log 2>&1 &
        #--noise_optimize \
        #--progressive_loss_mode \
        #--fixed_noise > exp21.log 2>&1 &

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/scratch/p200177/DE_371/inversion_experiments/exp22/inversion/' \
        --pack_dir='/project/scratch/p200177/DE_371/inversion_experiments/exp22/pack/' \
        --device='cuda:1' \
        --lambda_pixel=1 \
        --lambda_vgg=0 \
        --vgg_computation='sol3' \
        --lambda_noise=0 \
        --invstep=2000 \
        --inv_checkpoints='[250,500,1000,1500,2000]' \
        --date_start=2021-07-01 \
        --date_stop=2021-07-17 \
        --leadtimes='[3,6,9,12,15,18,21,24,27,30,33,36,39,42]' \
        --vgg_state_dict_path='/project/scratch/p200177/DE_371/resources/vgg_weights/vgg16-random.pth' \
        --plot_checkpoint > exp22.log 2>&1 &
        #--noise_optimize \
        #--progressive_loss_mode \
        #--fixed_noise > exp22.log 2>&1 &

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/scratch/p200177/DE_371/inversion_experiments/exp23/inversion/' \
        --pack_dir='/project/scratch/p200177/DE_371/inversion_experiments/exp23/pack/' \
        --device='cuda:2' \
        --lambda_pixel=1 \
        --lambda_vgg=1 \
        --vgg_computation='sol3' \
        --lambda_noise=0 \
        --invstep=2000 \
        --inv_checkpoints='[250,500,1000,1500,2000]' \
        --date_start=2021-07-01 \
        --date_stop=2021-07-17 \
        --leadtimes='[3,6,9,12,15,18,21,24,27,30,33,36,39,42]' \
        --vgg_state_dict_path='/project/scratch/p200177/DE_371/resources/vgg_weights/vgg16-random.pth' \
        --plot_checkpoint > exp23.log 2>&1 &
        #--noise_optimize \
        #--progressive_loss_mode \
        #--fixed_noise > exp23.log 2>&1 &

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/scratch/p200177/DE_371/inversion_experiments/exp24/inversion/' \
        --pack_dir='/project/scratch/p200177/DE_371/inversion_experiments/exp24/pack/' \
        --device='cuda:3' \
        --lambda_pixel=10 \
        --lambda_vgg=1 \
        --vgg_computation='sol3' \
        --lambda_noise=0 \
        --invstep=2000 \
        --inv_checkpoints='[250,500,1000,1500,2000]' \
        --date_start=2021-07-01 \
        --date_stop=2021-07-17 \
        --leadtimes='[3,6,9,12,15,18,21,24,27,30,33,36,39,42]' \
        --vgg_state_dict_path='/project/scratch/p200177/DE_371/resources/vgg_weights/vgg16-random.pth' \
        --plot_checkpoint > exp24.log 2>&1 &
        #--noise_optimize \
        #--progressive_loss_mode \
        #--fixed_noise > exp24.log 2>&1 &

wait
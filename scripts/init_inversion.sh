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


apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion.py \
        --ckpt_dir='/project/scratch/p200177/DE_371/victorsanchez/models/trained_generator/000024.pt' \
        --real_data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --output_dir='/project/scratch/p200177/DE_371/victorsanchez/results/member_inversion/optimization_inversion/test_optim_latent_fea13_rd/inversion/' \
        --pack_dir='' \
        --device='cuda' \
        --network_type='vgg16' \
        --lambda_perceptual_loss=1 \
        --channel_computation='sol2' \
        --invstep=1000 \
        --inv_checkpoints='[250,500,750,1000]' \
        --date_start=2021-07-01 \
        --date_stop=2021-07-02 \
        --leadtimes='[3]' \
        --plot_checkpoint \
        --features_after_relu \
        --feature_layers='[0,1,2,3]' \
        --feature_optimize \
        --feature_id=13 \

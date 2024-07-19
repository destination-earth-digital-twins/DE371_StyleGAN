#!/bin/bash -l
#SBATCH -J DE371_stylegan
#SBATCH -A p200177
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -p gpu
#SBATCH --ntasks-per-node=4
#SBATCH --time=03:00:00
#SBATCH --qos=short

# Note : to launch a train, choose the following parameters : --time=48:00:00 --qos=default
export TORCH_DISTRIBUTED_DEBUG=INFO 
export OMP_NUM_THREADS=4
export CUDA_HOME=/usr/local/cuda-12.1
export NVHPC_CUDA_HOME=/usr/local/cuda-12.1
export APPTAINERENV_CUDA_VISIBLE_DEVICES='0,1,2,3'
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export CXX=g++ #the compiler for cpp extensions
export CC=gcc  #the compiler to access the good cpp standard
export APPTAINER_BINDPATH="/project/home/p200177/DE_371/datasets:/project/home/p200177/DE_371/datasets/,/project/scratch/p200177/DE_371/victorsanchez:/project/scratch/p200177/DE_371/victorsanchez/"
export NCCL_ASYNC_ERROR_HANDLING=1
module load Apptainer/1.2.4-GCCcore-12.3.0

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif torchrun --nproc_per_node=4 main_gan.py \
        --data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --id_file="Large_lt_train_labels_1.csv" \
        --output_dir='/project/scratch/p200177/DE_371/victorsanchez/results/gan_training/test_1GPU/' \
        --g_channels=3 \
        --d_channels=3 \
        --epochs_num=30 \
        --var_names=['u','v','t2m'] \
        --config_dir='/project/scratch/p200177/DE_371/victorsanchez/results/gan_training/Set_UseNoiseFalse/' \
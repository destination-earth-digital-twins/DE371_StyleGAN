#!/bin/bash -l
#SBATCH -J DE371_stylegan
#SBATCH -A p200177
#SBATCH -q default
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00

export OMP_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda-12.1
export NVHPC_CUDA_HOME=/usr/local/cuda-12.1
export CXX=g++ #the compiler for cpp extensions
export CC=gcc  #the compiler to access the good cpp standard
export APPTAINER_BINDPATH="/project/home/p200177/DE_371/datasets:/project/home/p200177/DE_371/datasets/,/project/scratch/p200177/DE_371:/project/scratch/p200177/DE_371/"
module load env/release/2023.1
module load env/staging/2023.1
module load Apptainer/1.2.4-GCCcore-12.3.0


apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container_quantile_loss.sif  python3  main.py \
                --method_type='merge_into_giga_file'\
                --origin_csv='Large_lt_labels.csv'\
                --giga_directory='data/giga_test'\
                --main_path='/project/home/p200177/DE_371/datasets/dataset_Meteo_France_rr_u_v_t2m/data'\
                --data_directory='IS_rr_debug_1_1.0_0_0_0_0_0_256_large_lt'

#!/bin/bash -l
#SBATCH -J DE371_stylegan
#SBATCH -A p200177
#SBATCH -q test
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00

export OMP_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda-12.1
export NVHPC_CUDA_HOME=/usr/local/cuda-12.1
export CXX=g++ #the compiler for cpp extensions
export CC=gcc  #the compiler to access the good cpp standard
export APPTAINER_BINDPATH="/project/home/p200177/DE_371/:/project/home/p200177/DE_371/"
module load Apptainer/1.2.4-GCCcore-12.3.0

apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container.sif python3 temporal_interpolation.py \
    --device='cuda:0' \
    --num_workers=16 \
    --model_name='LatentInterpolatorCorrector2' \
    --model_path='interpolation/2025-01-08/model-1/LatentInterpolatorCorrector2-1024-3-pixel1000-epoch-20-2024-12-30T15_19.pt' \
    --num_layers=3 \
    --num_neurons=1024 \
    --normalization="Layer" \
    --dropout=0.0 \
    --base_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/' \
    --output_dir='interpolation/2025-01-08/model-1/' \
    --inv_dir='inversion_october/inversion/' \
    --pack_dir='inversion_october/pack/' \
    --start_date=2021-10-01 \
    --end_date=2021-11-01 > interpolation-1.log 2>&1
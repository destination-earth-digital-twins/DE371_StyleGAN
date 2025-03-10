#!/bin/bash -l
#SBATCH -J DE371_stylegan
#SBATCH -A p200177
#SBATCH -q default
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda-12.1
export NVHPC_CUDA_HOME=/usr/local/cuda-12.1
export CXX=g++ #the compiler for cpp extensions
export CC=gcc  #the compiler to access the good cpp standard
export APPTAINER_BINDPATH="/project/home/p200177/DE_371/:/project/home/p200177/DE_371/"
module load env/release/2023.1
module load env/staging/2023.1
module load Apptainer/1.2.4-GCCcore-12.3.0

apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container.sif python3 temporal_interpolation.py \
    --device='cuda:0' \
    --num_workers=16 \
    --model_name='StyleVectorInterpolator' \
    --model_path='interpolation_models/2025-01-21/StyleVectorInterpolator-512-4-perc50-pixel-250-lat025-epoch-20-2025-01-22T01_00.pt' \
    --num_layers=4 \
    --num_neurons=512 \
    --normalization="Layer" \
    --dropout=0.0 \
    --base_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/' \
    --output_dir='interpolation/2025-02-17/training-5/' \
    --inv_dir='inversion_test/inversion/' \
    --pack_dir='inversion_test/pack/' \
    --start_date=2021-06-16 \
    --end_date=2021-11-14 > interpolation-5.log 2>&1 &

apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container.sif python3 temporal_interpolation.py \
    --device='cuda:1' \
    --num_workers=16 \
    --model_name='StyleVectorInterpolatorCorrector' \
    --model_path='interpolation_models/2025-01-21/StyleVectorInterpolatorCorrector-512-4-mae1000-epoch-10-2025-02-04T01_04.pt' \
    --num_layers=4 \
    --num_neurons=512 \
    --normalization="Layer" \
    --dropout=0.0 \
    --base_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/' \
    --output_dir='interpolation/2025-02-17/training-6/' \
    --inv_dir='inversion_test/inversion/' \
    --pack_dir='inversion_test/pack/' \
    --start_date=2021-06-16 \
    --end_date=2021-11-14 > interpolation-6.log 2>&1 &
wait
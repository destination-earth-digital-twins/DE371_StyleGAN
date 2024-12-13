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
    --model_name='LatentInterpolatorCorrector' \
    --model_path='interpolation_models/2024-12-12/LatentInterpolatorCorrector-1024-3-pixel1000-2020-2021-period-epoch-20-2024-12-12T01_08.pt' \
    --num_layers=3 \
    --num_neurons=1024 \
    --normalization="Layer" \
    --dropout=0.0 \
    --base_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/' \
    --output_dir='interpolation/2024-12-12/pixel1000/' \
    --inv_dir='inversion_october/inversion/' \
    --pack_dir='inversion_october/pack/' \
    --start_date=2021-10-01 \
    --end_date=2021-10-07 > interpolation-1.log 2>&1 &

apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container.sif python3 temporal_interpolation.py \
    --device='cuda:1' \
    --num_workers=16 \
    --model_name='LatentInterpolatorCorrector' \
    --model_path='interpolation_models/2024-12-12/LatentInterpolatorCorrector-1024-3-pixel500-perc50-2020-2021-period-epoch-20-2024-12-12T05_23.pt' \
    --num_layers=3 \
    --num_neurons=1024 \
    --normalization="Layer" \
    --dropout=0.0 \
    --base_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/' \
    --output_dir='interpolation/2024-12-12/pixel500-perc50/' \
    --inv_dir='inversion_october/inversion/' \
    --pack_dir='inversion_october/pack/' \
    --start_date=2021-10-01 \
    --end_date=2021-10-07 > interpolation-2.log 2>&1 &

apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container.sif python3 temporal_interpolation.py \
    --device='cuda:2' \
    --num_workers=16 \
    --model_name='LatentInterpolatorCorrector' \
    --model_path='interpolation_models/2024-12-12/LatentInterpolatorCorrector-1024-3-pixel0-perc100-2020-2021-period-epoch-20-2024-12-12T05_26.pt' \
    --num_layers=3 \
    --num_neurons=1024 \
    --normalization="Layer" \
    --dropout=0.0 \
    --base_dir='/project/home/p200177/DE_371/experiments_WP2/temporal_downscaling_experiments/' \
    --output_dir='interpolation/2024-12-12/perc100/' \
    --inv_dir='inversion_october/inversion/' \
    --pack_dir='inversion_october/pack/' \
    --start_date=2021-10-01 \
    --end_date=2021-10-07 > interpolation-3.log 2>&1 &

wait
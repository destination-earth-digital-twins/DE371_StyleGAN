#!/bin/bash -l
#SBATCH -J DE371_stylegan
#SBATCH -A p200177
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -p gpu
#SBATCH --ntasks-per-node=4
<<<<<<< Updated upstream
<<<<<<< Updated upstream
#SBATCH --time=00:10
=======
#SBATCH --time=01:00:00
>>>>>>> Stashed changes
=======
#SBATCH --time=01:00:00
>>>>>>> Stashed changes
#SBATCH --qos=short

# Note : to launch a train, choose the following parameters : --time=48:00:00 --qos=default
export TORCH_DISTRIBUTED_DEBUG=INFO 
export OMP_NUM_THREADS=4
export CUDA_HOME=/usr/local/cuda-12.1
export NVHPC_CUDA_HOME=/usr/local/cuda-12.1
export CXX=g++ #the compiler for cpp extensions
export CC=gcc  #the compiler to access the good cpp standard
<<<<<<< Updated upstream
<<<<<<< Updated upstream
export APPTAINER_BINDPATH="/project/home/p200177/DE_371/datasets:/project/home/p200177/DE_371/datasets/,/project/scratch/p200177/DE_371/victorsanchez:/project/scratch/p200177/DE_371/victorsanchez/"
=======
export APPTAINER_BINDPATH="/project/home/p200177/DE_371/datasets:/project/home/p200177/DE_371/datasets/,/project/scratch/p200177/DE_371:/project/scratch/p200177/DE_371/"
>>>>>>> Stashed changes
=======
export APPTAINER_BINDPATH="/project/home/p200177/DE_371/datasets:/project/home/p200177/DE_371/datasets/,/project/scratch/p200177/DE_371:/project/scratch/p200177/DE_371/"
>>>>>>> Stashed changes
export NCCL_ASYNC_ERROR_HANDLING=1
module load Apptainer/1.2.4-GCCcore-12.3.0
module load NVHPC
module load GCC

<<<<<<< Updated upstream
<<<<<<< Updated upstream
apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif torchrun --nproc_per_node=4 main_gan.py
=======
apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif torchrun  --nproc_per_node=4 main_gan.py \
            
>>>>>>> Stashed changes
=======
apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif torchrun  --nproc_per_node=4 main_gan.py \
            
>>>>>>> Stashed changes

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

# Note : to launch a train, choose the following parameters : --time=48:00:00 --qos=default
export TORCH_DISTRIBUTED_DEBUG=INFO 
export OMP_NUM_THREADS=4
export CUDA_HOME=/usr/local/cuda-12.1
export NVHPC_CUDA_HOME=/usr/local/cuda-12.1
export APPTAINERENV_CUDA_VISIBLE_DEVICES='0,1,2,3'
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export CXX=g++ #the compiler for cpp extensions
export CC=gcc  #the compiler to access the good cpp standard
export APPTAINER_BINDPATH="/project/home/p200177/DE_371:/project/home/p200177/DE_371/,/project/scratch/p200177/DE_371:/project/scratch/p200177/DE_371/"
export NCCL_ASYNC_ERROR_HANDLING=1
module load Apptainer/1.2.4-GCCcore-12.3.0

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif torchrun --nproc_per_node=4 main_gan.py \
        --data_dir='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/' \
        --id_file="Large_lt_train_labels_1.csv" \
        --output_dir='/project/scratch/p200177/DE_371/victorsanchez/results/gan_training/exp20' \
        --batch_size=8 \
        --g_channels=45 \
        --d_channels=45 \
        --epochs_num=40000 \
        --var_names=['u','v','t2m'] \
        --config_dir='/project/scratch/p200177/DE_371/victorsanchez/results/gan_training/Set_UseNoiseFalse/' \
        --use_noise=True \
        --multi_timestep_mode \
        --nb_timesteps=15 \
        --timestep_period=3 \
        --stack_sample_along_time_and_variable \
        --channel_multiplier=10 \
        --pretrained_model=-1 \
        --sample_num=5 \
        --plot_samples=5 \
        --cutoff_dataset_leadtimes
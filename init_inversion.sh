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
module load Apptainer/1.2.4-GCCcore-12.3.0

apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion_precip_scenarios.py \
        --pixel_loss_type='mse'\
        --invstep=2000\
        --device='cuda:0'\
        --output_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/TEST_VGG_trained_sol5/mse/inversion/'\
        --pack_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/TEST_VGG_trained_sol5/mse/pack/'\
        #--noise_optimize\
        --vgg_computation='sol5'



# apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion_precip_scenarios.py \
#         --pixel_loss_type='mse'\
#         --invstep=2000\
#         --device='cuda:0'\
#         --output_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/batch1/mse/inversion/'\
#         --pack_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/batch1/mse/pack/'\
#         --real_data_dir='/project/scratch/p200177/DE_371/angeliquebonamy/data_basile_inv/samples_AROME_for_AE_1/batch1/'


# apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion_precip_scenarios.py \
#         --pixel_loss_type='mae'\
#         --invstep=2000\
#         --device='cuda:1'\
#         --output_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/batch1/mae/inversion/'\
#         --pack_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/batch1/mae/pack/'\
#         --real_data_dir='/project/scratch/p200177/DE_371/angeliquebonamy/data_basile_inv/samples_AROME_for_AE_1/batch1/'


# apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion_precip_scenarios.py \
#         --pixel_loss_type='wmse'\
#         --invstep=2000\
#         --device='cuda:2'\
#         --output_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/batch1/wmse/inversion/'\
#         --pack_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/batch1/wmse/pack/'\
#         --real_data_dir='/project/scratch/p200177/DE_371/angeliquebonamy/data_basile_inv/samples_AROME_for_AE_1/batch1/'


# apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion_precip_scenarios.py \
#         --pixel_loss_type='wamse'\
#         --invstep=2000\
#         --device='cuda:3'\
#         --output_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/batch1/wamse/inversion/'\
#         --pack_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/batch1/wamse/pack/'\
#         --real_data_dir='/project/scratch/p200177/DE_371/angeliquebonamy/data_basile_inv/samples_AROME_for_AE_1/batch1/'


# apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion_precip_scenarios.py \
#         --pixel_loss_type='amse'\
#         --invstep=2000\
#         --device='cuda:4'\
#         --output_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/batch1/amse/inversion/'\
#         --pack_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/batch1/amse/pack/'\
#         --real_data_dir='/project/scratch/p200177/DE_371/angeliquebonamy/data_basile_inv/samples_AROME_for_AE_1/batch1/'



# apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion_precip_scenarios.py \
#         --pixel_loss_type='mul_pixel_loss_mse'\
#         --invstep=2000\
#         --device='cuda:5'\
#         --output_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/batch1/mul_pixel_loss_mse/inversion/'\
#         --pack_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/batch1/mul_pixel_loss_mse/pack/'\
#         --real_data_dir='/project/scratch/p200177/DE_371/angeliquebonamy/data_basile_inv/samples_AROME_for_AE_1/batch1/'

# apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion_precip_scenarios.py \
#         --pixel_loss_type='mul_pixel_loss_mse'\
#         --invstep=2000\
#         --device='cuda:6'\
#         --output_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/batch1/mul_pixel_loss_mae/inversion/'\
#         --pack_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/batch1/mul_pixel_loss_mae/pack/'\
#         --real_data_dir='/project/scratch/p200177/DE_371/angeliquebonamy/data_basile_inv/samples_AROME_for_AE_1/batch1/'
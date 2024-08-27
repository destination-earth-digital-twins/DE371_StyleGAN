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
        --invstep=1500\
        --device='cuda:0'\
        --output_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/BON/VGG_random/sol5/inversion/'\
        --pack_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/BON/VGG_random/sol5/pack/'\
        # --optimize_features_computation\
        --vgg_computation='sol5'\
        # --noise_optimize


#  ['mse', 'mae','wmse','amse','wamse','sum_pixel_loss','sum_pixel_loss_mae','mul_pixel_loss_mae','mul_pixel_loss_mse']
# apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion_precip_scenarios.py \
#         --pixel_loss_type='mae'\
#         --invstep=1500\
#         --lambda_vgg=0\
#         --device='cuda:1'\
#         --output_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/mae/inversion/'\
#         --pack_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/mae/pack/'\
#      #   --optimize_features_computation\

# apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion_precip_scenarios.py \
#         --pixel_loss_type='wmse'\
#         --invstep=1500\
#         --lambda_vgg=0\
#         --device='cuda:2'\
#         --output_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/wmse/inversion/'\
#         --pack_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/wmse/pack/'\
#         # --optimize_features_computation\

# apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion_precip_scenarios.py \
#         --pixel_loss_type='amse'\
#         --lambda_vgg=0\
#         --invstep=1500\
#         --device='cuda:0'\
#         --output_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/amse/inversion/'\
#         --pack_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/amse/pack/'\
#         # --optimize_features_computation\

# apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion_precip_scenarios.py \
#         --pixel_loss_type='wamse'\
#         --lambda_vgg=0\
#         --invstep=1500\
#         --device='cuda:1'\
#         --output_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/wamse/inversion/'\
#         --pack_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/wamse/pack/'\
        # --optimize_features_computation\

# apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion_precip_scenarios.py \
#         --pixel_loss_type='mul_pixel_loss_mse'\
#         --invstep=1500\
#         --device='cuda:2'\
#         --output_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/batchs/mul_pixel_loss_mse/inversion/'\
#         --pack_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/batchs/mul_pixel_loss_mse/pack/'\
#         --optimize_features_computation\

# apptainer exec --nv /project/scratch/p200177/DE_371/resources/apptainer_container/container.sif python3 main_inversion_precip_scenarios.py \
#         --pixel_loss_type='mul_pixel_loss_mae'\
#         --invstep=1500\
#         --device='cuda:0'\
#         --output_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/batchs/mul_pixel_loss_mae/inversion/'\
#         --pack_dir='/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/batchs/mul_pixel_loss_mae/pack/'\
#         --optimize_features_computation\




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
export APPTAINER_BINDPATH="/PATH/TO/datasets/,/PATH/TO/DE_371:/PATH/TO/DE_371/"
module load Apptainer/1.2.4-GCCcore-12.3.0


# Inversion with pixel loss 

apptainer exec --nv /PATH/TO/apptainer_container/container.sif python3 main_perturbation.py \
        
        ### Directory Paths
        --ckpt_dir='Path to the checkpoint directory containing the pre-trained StyleGAN model.' \ 
        --real_data_dir=' Path to the directory containing real data used for inversion. ' \
        --data_dir="Path to the data directory containing the inversed ensembles. " \
        --output_dir=' Path to the directory where the gan-enriched ensembles will be stored. ' \ 
        --pack_dir=' Path to the directory where the real normalized ensembles that are inverted are stored. ' \ 
        --mean_file=' File containing mean values for normalization.  ' \
        --max_file=' File containing max values for normalization. ' \ 
        --device=' Device to run the inversion on (e.g., 'cuda:0').' \  

        #### Perturbation Parameters
        --inv_step='Number of optimization iterations.  ' \
        --var_indices= " List of variable indices to invert (e.g., [0,1,2,3]). Highly dependant on the shape of the samples of the dataset." \
        --Shape= ' Size of the samples as a tuple (channels, height, width). ' \ 
        --N_samples=" Ensemble size of the generated ensembles. " \
        --sample_rule="Perturbation method used for generating new ensembles (options: 'random', 'normal', 'w', 'extrapolation')." \
        --style_indices="Which vectors of the latent code should be perturbed, default[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]" \
        --conditioning_members=" Number of members used to generate perturbed ensembles (Max = 16)." \
        
        #### Data Control for Perturbation
        --dates_file="CSV file containing dates for inversion.  " \
        --date_start="Start date for inversion in the format 'YYYY-MM-DD'." \
        --date_stop="Stop date for inversion in the format 'YYYY-MM-DD'.  " \
        --leadtimes="List of lead times for inversion.  "\




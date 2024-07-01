#!/bin/bash

lam_MSE=$1

lam_PIPS=$2

lam_SCAT=$3

lam_MOCO=$4

lam_WNORM=$5 


echo "Launching with lam_MSE $lam_MSE, lam_PIPS $lam_PIPS, lam_SCAT $lam_SCAT, lam_MOCO $lam_MOCO, lam_WNORM $lam_WNORM "
	
sbatch /home/mrmn/brochetc/restyle-encoder/scripts/run_ReStylePSP.slurm "--dataset_type=arome_encode|--exp_dir=/scratch/mrmn/brochetc/GAN_2D/psp4arome_expe/|--workers=8|--batch_size=8|--test_batch_size=8|--test_workers=8|--val_interval=5000|--save_interval=10000|--start_from_latent_avg|--lpips_lambda=${lam_PIPS}|--l2_lambda=${lam_MSE}|--w_norm_lambda=${lam_WNORM}|--moco_lambda=${lam_MOCO}|--scat_lambda=${lam_SCAT}|--id_lambda=0|--input_nc=6|--n_iters_per_batch=2|--output_size=128|--stylegan_weights=/scratch/mrmn/brochetc/GAN_2D/psp4arome_expe/285000.pt"
 	


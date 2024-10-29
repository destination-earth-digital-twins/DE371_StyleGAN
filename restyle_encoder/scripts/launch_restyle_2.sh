#!/bin/bash

sbatch run_ReStylePSP_2.slurm "--dataset_type=arome_encode|--exp_dir=/scratch/mrmn/brochetc/GAN_2D/psp4arome_expe/|--workers=8|--batch_size=8|--test_batch_size=8|--test_workers=8|--val_interval=5000|--save_interval=10000|--start_from_latent_avg|--lpips_lambda=1.0|--scat_lambda=10.0|--l2_lambda=1.0|--w_norm_lambda=0.00|--learning_rate=0.0004|--input_nc=6|--n_iters_per_batch=2|--output_size=128|--stylegan_weights=/scratch/mrmn/brochetc/GAN_2D/psp4arome_expe/285000.pt"

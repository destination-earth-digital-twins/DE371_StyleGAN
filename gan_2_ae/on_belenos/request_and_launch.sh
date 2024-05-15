#!/bin/bash
echo "$0 executed on $(hostname)"
ARGS=$1
ARGS2="${ARGS//|/ }"
cd SAVE
srun --partition=normal256 --job-name=pre_proc_for_AE --nodes=1 --ntasks=1 --cpus-per-task=1 --time=01:00:00 python3 -u pre_proc_for_AE.py $ARGS2

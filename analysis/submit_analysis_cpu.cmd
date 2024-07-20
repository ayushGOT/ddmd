#!/bin/bash
#SBATCH -J AN14RNA
#SBATCH -t 15:00:00
#SBATCH -N 1
#SBATCH -n 48
# SBATCH --ntasks-per-node=4
# SBATCH --cpus-per-task=7
# SBATCH --gpus=1
#SBATCH -A zerze

source activate /project/zerze/ddmd

#mkdir cpts
#cp dynamic.cpt cpts/dynamic_${SLURM_JOB_ID}.cpt

ddmd analysis -c infer.yml

conda deactivate


#!/bin/bash
#SBATCH -J AN13RNA
#SBATCH -t 20:00:00
#SBATCH -N 1
# SBATCH -n 40
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH -A zerze

source activate /project/zerze/ddmd

#mkdir cpts
#cp dynamic.cpt cpts/dynamic_${SLURM_JOB_ID}.cpt

ddmd analysis -c infer.yml

conda deactivate


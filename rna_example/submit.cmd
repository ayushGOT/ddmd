#!/bin/bash
#SBATCH -J 8RNAQ0.5
#SBATCH -t 168:00:00
#SBATCH -N 1
#SBATCH -n 48 
# SBATCH --ntasks-per-node=6
# SBATCH --cpus-per-task=8
#SBATCH --gpus=8
#SBATCH -A zerze

source activate /project/zerze/ddmd

#mkdir cpts
#cp dynamic.cpt cpts/dynamic_${SLURM_JOB_ID}.cpt

python ../run_ddmd.py -c simple.yml

conda deactivate

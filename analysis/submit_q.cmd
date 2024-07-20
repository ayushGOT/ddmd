#!/bin/bash
#SBATCH -J QddmdRNA
#SBATCH -t 96:00:00
#SBATCH -N 1
#SBATCH -n 40
# SBATCH --ntasks-per-node=6
# SBATCH --cpus-per-task=8
# SBATCH --gpus=8
#SBATCH -A zerze

#mkdir cpts
#cp dynamic.cpt cpts/dynamic_${SLURM_JOB_ID}.cpt

module load intel-oneapi
source ~/PROGRAMS/plumed2-v2.8/sourceme.sh
bash q.sh > out.out

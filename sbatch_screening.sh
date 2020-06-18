#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J Marij_Leg_Screening


# email error reports
#SBATCH --mail-user=nathaniel_goodman@brown.edu 
#SBATCH --mail-type=ALL

# output file
#SBATCH -J screening
#SBATCH -o screening-%j.out
#SBATCH -e screening-%j.err
# %A is the job id (as you find it when searching for your running / finished jobs on the cluster)
# %a is the array id of your current array job


# Request runtime, memory, cores
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH -c 32
#SBATCH -N 1
#SBATCH --batch_id=0-100

eval "$(conda shell.bash hook)"
conda activate marijuana_study

machine='ccv'

python Neural_Filtering.py --machine $machine 
#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script for Arc nodes
#SBATCH -J myjob # Job Name
#SBATCH -o myjob.o%j # Name of the stdout output file
#SBATCH -e myjob.e%j # Name of stderr error file
#SBATCH -p gpu4v100 --gres=gres/gpu:2 # Queue (partition) name
#SBATCH -N 1 # Total # of nodes (must be 1 for serial)
#SBATCH -n 5 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:30:00 # Run time (hh:mm:ss)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jayanta.dey@utsa.edu
#Put other commands such as module load if needed after this line (All
# commands must follow #SBATCH directives)

# Launch code...
module load anaconda3/2024.02-1 
conda activate contrastive
python cifar10_exp.py



#!/bin/bash -l
#SBATCH --time=72:00:00
#SBATCH --ntasks=20
#SBATCH --mem=300g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rosal072@umn.edu

module load python3/3.10.9_anaconda2023.03_libmamba


python /home/nboehnke/rosal072/CSCI5461_Project/NN/neural_networks.py



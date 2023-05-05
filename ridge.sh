#!/bin/bash -l
#SBATCH --time=17:00:00
#SBATCH --ntasks=20
#SBATCH --mem=200g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rosal072@umn.edu

module load python


python /home/nboehnke/rosal072/CSCI5461_Project/Ridge/ridge.py




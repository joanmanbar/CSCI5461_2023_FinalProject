#!/bin/bash -l
#SBATCH --time=73:00:00
#SBATCH --ntasks=20
#SBATCH --mem=350g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rosal072@umn.edu

module load python


python /home/nboehnke/rosal072/CSCI5461_Project/SVR/support_vector_regressor_cv.py



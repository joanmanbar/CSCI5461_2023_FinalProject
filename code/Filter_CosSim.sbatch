#!/bin/bash -l
#SBATCH --job-name=CosSim #
#SBATCH --time=04:00:00
#SBATCH --ntasks=128
#SBATCH --mem=500g
#SBATCH --tmp=500g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=jbarreto@umn.edu 
#SBATCH -e $J.error_CosSim
#SBATCH -o $J.output_CosSim
#SBATCH -p ag2tb


# Previous steps
# ssh -Y jbarreto@mesabi.msi.umn.edu
# cd /home/myersc/jbarreto/CSCI5461_Project

cd /home/myersc/jbarreto/CSCI5461_Project/code
module load python3/3.9.3_anaconda2021.11_mamba
python3 filter_dense_cossim.py
 

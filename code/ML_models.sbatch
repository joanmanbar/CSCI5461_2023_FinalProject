#!/bin/bash -l
#SBATCH -A myersc
#SBATCH --job-name=ML_test_lr #
#SBATCH --time=24:00:00
#SBATCH --ntasks=8
#SBATCH --mem=40g
#SBATCH --tmp=20g
#SBATCH --array=1-5 # 5 jobs
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=jbarreto@umn.edu 
#SBATCH -e error.$J
#SBATCH -o output.$J

# Previous steps
# ssh -Y jbarreto@mesabi.msi.umn.edu
# cd /home/myersc/jbarreto/CSCI5461_Project
# module load python3/3.9.3_anaconda2021.11_mamba
# pip install -r requirements.txt


newgrp myersc
cd /home/myersc/jbarreto/CSCI5461_Project/code

python3 ML_models.py


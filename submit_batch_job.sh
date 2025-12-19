#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --gres=gpu:v100:2 
#SBATCH --time=168:00:00
#SBATCH --mem=60gb
#SBATCH --job-name=OmniProt_c1
#SBATCH --error=OmniProt_c1.%J.err
#SBATCH --output=OmniProt_c1.out

# Change directory to the working directory
cd OmniProt

# Load necessary modules
module load anaconda/23
conda activate OmniProt
# run OmniProt pipeline.. could take 24-48hours for the LBP dataset... 
python run_pipeline_1.py
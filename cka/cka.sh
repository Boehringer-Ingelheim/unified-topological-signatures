#!/bin/bash
#SBATCH --job-name=cka
#SBATCH --output=cka.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=128GB

python run_cka.py 

 
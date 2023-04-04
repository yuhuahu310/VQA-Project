#!/bin/bash
#
#SBATCH --time=04:00:00
#SBATCH --job-name=128_4_eval
#SBATCH --chdir=/home/katehu/vqa/multimodal_baseline
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=threedle-owned
#SBATCH --gres=gpu:1
#SBATCH --nodelist=c001
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16G
#
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=katehu310@gmail.com

srun python -u test3.py


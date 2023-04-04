#!/bin/bash
#
#SBATCH --time=04:00:00
#SBATCH --job-name=vqa_128_6
#SBATCH --chdir=/home/katehu/vqa/multimodal_baseline
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=threedle-owned
#SBATCH --gres=gpu:1
#SBATCH --nodelist=f003
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16G
#
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=katehu310@gmail.com

srun python -u run3.py --hidden 128 --layers 6 --exp hidden128_layer6 --epoch 20


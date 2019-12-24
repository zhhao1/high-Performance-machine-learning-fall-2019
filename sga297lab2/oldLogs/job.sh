#!/bin/bash
#

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=150:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:p40:1
#SBATCH --job-name=gowri
#SBATCH --mail-type=END
#SBATCH --mail-user=sga297@nyu.edu
#SBATCH --output=gowri.out

source py3.6.3/bin/activate
python -u main.py --Epochs 5 --Gpu False --TrainNumWorkers 4 >> logc5.txt


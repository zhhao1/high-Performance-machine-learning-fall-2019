#!/bin/bash

source ../../py3.6.3/bin/activate

# c1
#python main.py  --Epochs 5 >> logC1.txt

# c2
#python main.py --Epochs 5 >> logC2.txt

# c3
#python main.py --TrainNumWorkers 0 --Epochs 5 >> log0C3.txt
#python main.py --TrainNumWorkers 4 --Epochs 5  >> log4C3.txt
#python main.py --TrainNumWorkers 8 --Epochs 5  >> log8C3.txt
#python main.py --TrainNumWorkers 12 --Epochs 5 >> log12C3.txt

# c4
#python main.py --TrainNumWorkers 1 --Epochs 5 >> log1c4.txt
#python main.py --TrainNumWorkers 4 --Epochs 5  >> log4c4.txt

# c5
#python main.py --Epochs 5 --TrainNumWorkers 4 >> log4c5GPU.txt
# for cpu run on local machine.

#c6
#python main.py --TrainNumWorkers 4 --Optimizer 'sgd' --Epochs 5 >> logc6sgd.txt
#python main.py --TrainNumWorkers 4 --Optimizer 'sgd_nes' --Epochs 5 >> logc6sgdnes.txt
#python main.py --TrainNumworkers 4 --Optimizer 'adagrad' --Epochs 5 >> logc6adagrad.txt
#python main.py --TrainNumWorkers 4 --Optimizer 'adadelta' --Epochs 5 >> logc6adadelta.txt
#python main.py --TrainNumWorkers 4 --Optimizer 'adam' --Epochs 5 >> logc6adam.txt

#c7
python mainBN.py --TrainNumWorkers 4 --Epochs 5 >> logc7BN.txt

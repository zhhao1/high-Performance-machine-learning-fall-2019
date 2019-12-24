#!/bin/bash
source ../../../py3.6.3/bin/activate

python main.py --batchtraining 2048 --gpus 4 >> log_c4_2048_fourgpus.txt
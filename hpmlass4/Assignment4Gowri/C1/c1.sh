#!/bin/bash
source ../../../py3.6.3/bin/activate
python main.py --batchtraining 32 --gpus 1 >> log_c1_32_onegpu.txt
python main.py --batchtraining 128 --gpus 1 >> log_c1_128_onegpu.txt
python main.py --batchtraining 512 --gpus 1 >> log_c1_512_onegpu.txt
python main.py --batchtraining 2048 --gpus 1 >> log_c1_2048_onegpu.txt
#!/bin/bash
source ../../../py3.6.3/bin/activate
python main.py --batchtraining 32 --gpus 1 >> log_c2_32_onegpu.txt
python main.py --batchtraining 128 --gpus 1 >> log_c2_128_onegpu.txt
python main.py --batchtraining 512 --gpus 1 >> log_c2_512_onegpu.txt

python main.py --batchtraining 64 --gpus 2 >> log_c2_64_twogpus.txt
python main.py --batchtraining 256 --gpus 2 >> log_c2_256_twogpus.txt
python main.py --batchtraining 1024 --gpus 2 >> log_c2_1024_twogpus.txt

python main.py --batchtraining 128 --gpus 4 >> log_c2_128_fourgpus.txt
python main.py --batchtraining 512 --gpus 4 >> log_c2_512_fourgpus.txt
python main.py --batchtraining 2048 --gpus 4 >> log_c2_2048_fourgpus.txt
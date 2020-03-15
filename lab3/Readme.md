Steps to run:

Run in pyenv.
module purge
module load cuda/9.0.176
module load cudnn/9.0v7.0.5
Compile and run with “nvcc -o lab3 lab3.cu -lcublas -lcudnn ; ./lab3”
Run experiments on Prince with flags –gres=gpu:p40:1 –mem 20GB

For any queries mail, sga297@nyu.edu

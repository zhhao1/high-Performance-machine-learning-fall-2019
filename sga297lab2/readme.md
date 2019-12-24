Readme File

Name: Sree Gowri Addepalli
Email: sga297@nyu.edu

1. Environment setup:
Follow these steps exactly as given to setup in prince:
• module purge
• module load python3/intel/3.6.3
• mkdir pyenv
• cd pyenv
• virtualenv --system-site-packages py3.6.3
• source ~/pyenv/py3.6.3/bin/activate
• pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36mlinux_x86_64.whl
• pip3 install torchvision
• pip3 install torchsummary
• copy the hpml folder inside pyenv directory.
• This internally has a folder named FCSubmission



2. Gpu Configuration:

1. Prince Cluster
2. GPU Used type: srun --gres=gpu:p1080:1 --pty /bin/bash   
(GPU-10, GPU-11, GPU-12) 

3. CPU Configuration:

The following Config is used to run C5 in CPU mode. (See CPU specs image). For running on CPU, use the main.py as mentioned in the CPU folder within FCSubmission.

Inside the folder 'FCSubmission':


Run the below commands to generate all the log files:
chmod +x hpml2Run.sh
 ./hpml2Run.sh
 
 
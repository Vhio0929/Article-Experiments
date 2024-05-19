import os

cmd = 'nohup python -u train_cifar10_16.py  >trian_ori18.out 2>&1 &'
os.system(cmd)
cmd = 'CUDA_VISIBLE_DEVICES=1 nohup python -u train_LS_16.py  >train_LS16.out 2>&1 &'
os.system(cmd)




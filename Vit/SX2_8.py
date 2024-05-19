import os

cmd = 'nohup python -u train_cifar10.py  >4train_LS_8.out 2>&1 &'
os.system(cmd)
cmd = 'nohup python -u train_cifar10_4.py  >4train_LS_4.out 2>&1 &'
os.system(cmd)
cmd = 'nohup python -u train_cifar10_16.py  >4train_LS_16.out 2>&1 &'
os.system(cmd)


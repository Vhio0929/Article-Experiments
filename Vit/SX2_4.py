import os

cmd = 'nohup python -u train_cifar10.py  >4train_ori_patch1.out 2>&1 &'
os.system(cmd)
cmd = 'nohup python -u train_LS.py  >4train_LS_1_patch.out 2>&1 &'
os.system(cmd)




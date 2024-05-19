import os

cmd = 'nohup python -u train2.py  >ori4_trian.out 2>&1 &'
os.system(cmd)
cmd = 'CUDA_VISIBLE_DEVICES=1 nohup python -u train6.py  >LSHAM4_train6.out 2>&1 &'
os.system(cmd)




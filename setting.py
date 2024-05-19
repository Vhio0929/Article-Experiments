"""
Created bt tz on 2020/11/12 
"""

__author__ = 'tz'

import torch

# UNK = 0
# PAD = 1
UNK = 1
PAD = 0
BATCH_SIZE = 64
TRAIN_FILE = 'data/train.txt'  # 训练集
DEV_FILE = 'data/dev.txt'  # 验证集
TEST_FILE = 'data/test.txt'  # 测试文件
SAVE_FILE = 'save/5_1_de_en/jf_8_de_en_model.pt'  # 模型保存路径(注意如当前目录无save文件夹需要自己创建)
LAYERS = 6  # encoder和decoder层数
D_MODEL = 512  # embedding 维度
D_FF = 1024  # feed forward第一个全连接层维数
H_NUM = 4  # multi head attention hidden个数
DROPOUT = 0.1  # dropout比例
EPOCHS = 200
MAX_LENGTH = 60

SRC_VOCAB = 19213  # 英文的单词数
TGT_VOCAB = 9796  # 中文的单词数

#SRC_VOCAB = 9796  # 英文的单词数
#TGT_VOCAB = 19213  # 中文的单词数

# SRC_VOCAB = 5493  # 英文的单词数
# TGT_VOCAB = 3196  # 中文的单词数

# 这里针对的是DEV文件
BLEU_REFERENCES = "data/bleu/references.txt" # BLEU评价参考译文
BLEU_CANDIDATE = "data/bleu/candidate.txt"  # 模型翻译译文
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE='cpu'

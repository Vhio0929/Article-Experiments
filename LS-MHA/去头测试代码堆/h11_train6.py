"""
Created bt tz on 2020/11/12 
"""

__author__ = 'tz'

import torch
import time

from data_pre import PrepareData
from setting import EPOCHS
from model6 import SimpleLossCompute, LabelSmoothing, NoamOpt
from h11_model6 import make_model

from setting import DEVICE
import os
from torch.utils.tensorboard import SummaryWriter
 



print("====DEVICE====:", DEVICE)


from setting import LAYERS,D_MODEL,D_FF,DROPOUT,H_NUM,TGT_VOCAB,SRC_VOCAB,\
    SAVE_FILE,TRAIN_FILE,DEV_FILE
# 模型的初始化
model = make_model(
                    SRC_VOCAB,
                    TGT_VOCAB,
                    LAYERS,
                    D_MODEL,
                    D_FF,
                    H_NUM,
                    DROPOUT
                )
def run_epoch(data, model, loss_compute, epoch):
    """
    迭代一次数据集
    :param data:
    :param model:
    :param loss_compute: loss_compute函数
    :param epoch: 传入的迭代次数
    :return:
    """
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.

    for i, batch in enumerate(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens  # 实际的词数

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch %d Batch: %d Loss: %f Tokens per Sec: %fs" % (
            epoch, i - 1, loss / batch.ntokens, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


def train(data, model, criterion, optimizer):
    """
    训练并保存模型
    """
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_dev_loss = 1e5
    exit_num = 0
    for epoch in range(EPOCHS): # EPOCES在代码开头指定
        """
        每次迭代一次就在验证集上验证loss
        """
        print("epoch:", epoch)
        model.train()
        train_loss = run_epoch(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)

        train_writer.add_scalar('Train_loss', train_loss, epoch)
        model.eval()

        # 在dev集上进行loss评估
        print('>>>>> Evaluate')
        dev_loss = run_epoch(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        print('<<<<< Evaluate loss: %f' % dev_loss)
        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        exit_num += 1
        if dev_loss < best_dev_loss:
            exit_num = 0
            torch.save(model.state_dict(), 'save/LS_HAM.pt')
            best_dev_loss = dev_loss
            print('****** Save model done... ******')
        if epoch == 100:
            torch.save(model.state_dict(), 'save/LS_HAM_100.pt')
        if exit_num == 10:
            print("====超过10轮未增长，保存退出====")
            break
        print()
        test_writer.add_scalar('Dev_loss', dev_loss, epoch)

if __name__ == '__main__':

    
    # 将 train accuracy 保存到 "tensorboard/train" 文件夹
    log_dir = os.path.join('tensorboard16', 'LS_HAM_train')

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        print(f"{log_dir}创建成功！")
    train_writer = SummaryWriter(log_dir=log_dir)

    # 将 test accuracy 保存到 "tensorboard/zsy_test" 文件夹
    log_dir = os.path.join('tensorboard16', 'LS_HAM_dev')

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        print(f"{log_dir}创建成功！")
    test_writer = SummaryWriter(log_dir=log_dir)

    print('处理数据')
    data = PrepareData(TRAIN_FILE,DEV_FILE, sort=True)

    print('>>>开始训练')
    train_start = time.time()
    # 损失函数
    criterion = LabelSmoothing(TGT_VOCAB, padding_idx=0, smoothing=0.0)
    # 优化器
    optimizer = NoamOpt(D_MODEL, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    train(data, model, criterion, optimizer)
    print(f'<<<训练结束, 花费时间 {time.time() - train_start:.4f}秒')


    # 对测试数据集进行测试
    # print('开始测试')
    # from test import evaluate_test
    # evaluate_test(data, model)


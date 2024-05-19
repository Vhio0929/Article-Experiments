import torch
import torch.nn as nn

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from typing import Iterable, List
from torch import Tensor
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from timeit import default_timer as timer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 源语言是德语
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# 定义token字典，定义vocab字典
token_transform = {}
vocab_transform = {}

# 创建源语言和目标语言的tokenizer
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

# 构建生成分词的迭代器
def yield_tokens(data_iter:Iterable, language:str) -> List[str]:
    language_index = {SRC_LANGUAGE:0, TGT_LANGUAGE:1}

    for data_sample in data_iter:

        yield token_transform[language](data_sample[language_index[language]])

# 定义特殊字符
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

# 确保标记按其索引的顺序正确插入到词表中
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:

    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)






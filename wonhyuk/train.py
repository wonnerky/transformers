import numpy as np
import pandas as pd
import torch, os, tokenizers, string, re, json, requests
from torch.nn import functional as F
import torch.nn as nn
from tqdm import tqdm
from transformers import BartTokenizer, DataCollatorForLanguageModeling, Trainer
# from wonhyuk.build_edited_bart_model import init_edited_bart_model
from wonhyuk.data_processing import WikiDataset

MAX_LEN = 192
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 5

TOKENIZER = BartTokenizer.from_pretrained('bart-large')

# model = init_edited_bart_model('gen')

'''
dataset
        data transform into tensor
        output
            type : tensor list      
            tensor([   0,   91,   56,   10, 4910,  774,   11,    5, 2384,  651, 3052,  610,
                   926,  196,   11, 5241,    4,    2])
'''
dataset = WikiDataset(
    tokenizer=TOKENIZER,
    block_size=128,
    data_type='test',
    flag=1
)


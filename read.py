#简单验证数据是否正确处理
import os
import json
import numpy as np
import argparse
from tqdm import tqdm
import sys
sys.path.append("..")
from tokenizations import tokenization_bert

def dele(data):
    for i in data[::-1]:
        if i == '$$$':
            data.remove(i)
    return data

txt_num = 1
tokenizer = tokenization_bert.BertTokenizer('cache/vocab_small.txt')
for i in range(txt_num):
    data = np.load('/home/chenyu/可训练格式处理/gpt3_dataset/weixin_tokenized/text_1.npy')#在此填入要计算的数据集的文件夹地址
    # print(data[0][:3000][:100])
    text = tokenizer.convert_ids_to_tokens(data[0][:3000])
    # print(len(data))
    # for k in range(len(text)-1):
    #     if '##' in text[k+1]:
    #         text[k+1] = "".join((text[k]+text[k+1]).split('##'))
    #         text[k] = '$$$'
    # text = dele(text)
    # print(" ".join(text),str(i))
    print("".join(text),str(i))

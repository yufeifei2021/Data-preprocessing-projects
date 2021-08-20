import os
import json
import numpy as np
import argparse
from tqdm import tqdm
import sys
from tokenizations import tokenization_bert
import pathlib
import random


def build_files(
    data_path,
    tokenized_data_path,
    num_pieces,
    full_tokenizer,
    window_size,
    stride,
    model
):

    # with open(data_path, "r", encoding="utf8") as f:
    #     print("reading lines")
    #     lines = json.load(f)
    lines =  open(data_path, "r", encoding="utf8").readlines()
    all_len = len(lines)
    # print(all_len)
    # exit(0)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)

    print("begin")
    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * i : all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(lines[all_len // num_pieces * (i + 1) :])
        if model == "normal":
            sublines = [
                full_tokenizer.tokenize(line) for line in sublines #if len(line) > min_length
            ]
            sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
            full_line = []
            for subline in sublines:
                full_line.append(
                    full_tokenizer.convert_tokens_to_ids("[MASK]")
                )  # 文章开头添加MASK表示文章开始
                full_line.extend(subline)
                full_line.append(
                    full_tokenizer.convert_tokens_to_ids("[CLS]")
                )  # 文章之间添加CLS表示文章结束
        elif model == "QA":
            sublines = [
                full_tokenizer.tokenize(mini_line) for line in sublines for mini_line in line #if len(mini_line) > min_length
            ]
            sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
            full_line = []
            ############
            #不同QA类型数据集修改此处
            for idx in range(0,len(sublines),2):#list的长度
                full_line.append(
                    full_tokenizer.convert_tokens_to_ids("[MASK]")
                )  # 文章开头添加MASK表示文章开始
                # full_line.append(
                #     full_tokenizer.convert_tokens_to_ids("[C]")
                # )  # 在问题前加标签Q
                full_line.extend(sublines[idx])
                # full_line.append(
                #     full_tokenizer.convert_tokens_to_ids("[E]")
                # )  # 在回答前加标签A
                full_line.append(
                    97
                )  # 在问题前加标签Q
                full_line.extend(sublines[idx+1])
                full_line.append(
                    full_tokenizer.convert_tokens_to_ids("[CLS]")
                )  # 文章之间添加CLS表示文章结束
                ############
        to_text = full_line[:-1]
        to_label = full_line[1:]
        text = []
        label = []
        # print(len(to_text))
        for j in range(0, len(to_text), stride):
            # print(j)
            if len(to_text[j : j + window_size]) < window_size:
                # print("@@@@@@@@@@@@@@@")
                text.append(
                    np.asarray(
                        to_text[j : j + window_size]
                        + [full_tokenizer.convert_tokens_to_ids("[CLS]")]
                        * (window_size - len(to_text[j : j + window_size]))
                    )
                )
                label.append(
                    np.asarray(
                        to_label[j : j + window_size]
                        + [full_tokenizer.convert_tokens_to_ids("[CLS]")]
                        * (window_size - len(to_text[j : j + window_size]))
                    )
                )
            else:
                text.append(np.asarray(to_text[j : j + window_size]))
                label.append(np.asarray(to_label[j : j + window_size]))
        text = np.asarray(text)
        label = np.asarray(label)
        np.save(tokenized_data_path + "text_{}".format(i+1), text)
        np.save(tokenized_data_path + "label_{}".format(i+1), label)
    print("finish")


def get_stride(file):
    data = open(file,'r', encoding="utf8").readlines()
    len_data = len(data)
    resultList = random.sample(range(0, len_data), 20)
    max = 0
    strd = 0
    for i in resultList:
        temp = len(data[i])
        # print(temp)
        if temp > max:
            max = temp
    if max >1000:
        strd = 896
    elif 300 < max <= 1000:
        strd = 960
    else:
        strd = 992
    return strd

def get_token_num(dir):
    p = pathlib.Path(dir) 
    nums = 0
    for f in p.rglob('text*'):
        if f.is_file():
            data = np.load(f)
            nums = nums + len(data)*1024
    print(f'本次处理的token数是{nums}')

def main(args):  # 暂时只考虑输入大于等于窗口的情况
    sizes = os.path.getsize(args.raw_data_path)/1024/1024/1024
    print(f'文件大小为：{sizes}G')
    args.num_pieces = int(sizes//3) +1
    print(f'分成{args.num_pieces}份')
    args.stride = get_stride(args.raw_data_path)
    print(f'步长选择{args.stride}')
    full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    print("building files")
    build_files(
                data_path=args.raw_data_path,
                # tokenized_data_path=f"./gpt3_dataset/zhuanli_tokenized_{index}/",
                tokenized_data_path=args.tokenized_data_path,
                num_pieces=args.num_pieces,
                full_tokenizer=full_tokenizer,
                window_size=args.window_size,
                stride=args.stride,
                model=args.model
            )
    # p = pathlib.Path(args.raw_data_path)
    # index = 6
    # # for f in p.rglob('*'):
    # #     if f.is_file():
    # #         print(f)
    # # exit(0)
    # for f in p.rglob('*'):
    #     if f.is_file():
    #         build_files(
    #             data_path=f,
    #             # tokenized_data_path=f"./gpt3_dataset/zhuanli_tokenized_{index}/",
    #             tokenized_data_path=args.tokenized_data_path
    #             num_pieces=args.num_pieces,
    #             full_tokenizer=full_tokenizer,
    #             window_size=args.window_size,
    #             stride=args.stride,
    #             model=args.model
    #         )
    #         index = index + 1
    print("files built")
    get_token_num(args.tokenized_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        default="/home/chenyu/可训练格式处理/fsdownload/law/4.txt",
        type=str,
        required=False,
        help="原始训练语料",
    )
    parser.add_argument(
        "--tokenized_data_path",
        default="./gpt3_dataset/law_tokenized(4)/",
        type=str,
        required=False,
        help="tokenized语料存放位置",
    )
    parser.add_argument(
        "--num_pieces", default=1, type=int, required=False, help="将训练语料分成多少份"
    )
    parser.add_argument(
        "--tokenizer_path",
        default="cache/vocab_small.txt",
        type=str,
        required=False,
        help="选择词库",
    )
    parser.add_argument(
        "--model",
        default="normal",#QA
        type=str,
        required=False,
        help="txt type",
    )
    parser.add_argument(
        "--window_size", default=1024, type=int, required=False, help="窗口大小"
    )
    parser.add_argument(
        "--stride", default=768, type=int, required=False, help="训练时取训练数据的窗口步长"
    )
    args = parser.parse_args()
    main(args)



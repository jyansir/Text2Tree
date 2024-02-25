from transformers import AutoTokenizer
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import json
from collections import defaultdict

np.random.seed(7)

def gen_taxonomy(data_path):
    with open(f'{data_path}/all_label2id.json', 'r') as f:
        label2id = json.load(f)
    label_list = list(label2id.keys())
    first_classes = [c for c in label_list if len(c) == 1]
    second_classes = [c for c in label_list if len(c) == 3]
    third_classes = [c for c in label_list if '.' in c]

    lines = ''
    lines += '\t'.join(['Root'] + first_classes) + '\n'
    for fc in first_classes:
        lines += '\t'.join([fc] + [c for c in second_classes if c.startswith(fc)]) + '\n'
    for sc in second_classes:
        lines += '\t'.join([sc] + [c for c in third_classes if c.startswith(sc)]) + '\n'
    with open(f'{data_path}/taxonomy', 'w') as f:
        f.write(lines)

def gen_hierarchy(model_name, data_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label_dict = {}
    with open(f'{data_path}/all_label2text.json', 'r') as f:
        label2text = json.load(f)
    label_dict.pop
    hiera = defaultdict(set)
    taxonomy_file = f'{data_path}/taxonomy'
    if not os.path.exists(taxonomy_file):
        print('generating taxonomy hierarchy')
        gen_taxonomy(data_path)
    with open(taxonomy_file, 'r') as f:
        label_dict['Root'] = -1
        for line in f.readlines():
            line = line.strip().split('\t')
            for i in line[1:]:
                if i not in label_dict:
                    label_dict[i] = len(label_dict) - 1
                hiera[label_dict[line[0]]].add(label_dict[i])
        label_dict.pop('Root')
        hiera.pop(-1)
    value_dict = {i: tokenizer.encode(label2text[v], add_special_tokens=False) for v, i in label_dict.items()}
    print('saving `bert_value_dict.pt` and `label_dict.json` at: ', data_path)
    torch.save(value_dict, f'{data_path}/bert_value_dict.pt')
    torch.save(hiera, f'{data_path}/slot.pt')
    with open(f'{data_path}/label_dict.json', 'w') as f:
        json.dump(label_dict, f, indent=4)

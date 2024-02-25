import sys
import json
import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

sys.path.append('.')
from data_utils import init_data_handler

def convert_hpt_data(dataset):
    @dataclass
    class DataArgs:
        dataset: str
    data_args = DataArgs('pubmed_multilabel')
    Xs, id2label = init_data_handler(None, data_args,return_raw=True)

    parent_labels = sorted(list(set([v.split('.')[0] for v in id2label.values()])))
    child_labels = list(id2label.values())

    valuedict = {i: v for i, v in enumerate(parent_labels + child_labels)}
    v2i = {v:k for k,v in valuedict.items()}

    with open(f'./data/{dataset}/all_label2text.json', 'r') as f:
        label2text = json.load(f)
    new_valuedict = {i: label2text[v] for i, v in valuedict.items()}

    torch.save(new_valuedict, f"./data/{dataset}/hpt_value_dict.pt")

    # parent node (label) number (non-leaf labels)
    with open(f"./data/{dataset}/hpt_data_info.json", 'w') as f:
        saved_infos = {'num_parent_nodes': len(new_valuedict) - len(id2label)}
        json.dump(saved_infos, f, indent=4)

    slot2value = {}
    for v, i in v2i.items():
        if '.' not in v and i not in slot2value:
            slot2value[i] = set()
        else:
            parent = v.split('.')[0]
            parent_i = v2i[parent]
            if parent_i not in slot2value:
                slot2value[parent_i] = set()
            slot2value[parent_i].add(i)
        
    slot2value = defaultdict(set, slot2value)
    torch.save(slot2value, f"./data/{dataset}/hpt_slot.pt")

    value2id = {v: k for k, v in valuedict.items()}

    def build_htc_label(label):
        label_ids = []
        if isinstance(label, np.ndarray):
            label_ids = np.where(label == 1)[0].tolist()
        elif isinstance(label, np.integer):
            label_ids = [label]
        htc_labels = []
        label_names = []
        for i in label_ids:
            v = id2label[i]
            label_names.append(v)
            p = v.split('.')[0]
            if p not in htc_labels:
                htc_labels.append(p)
        htc_labels += label_names
        htc_ids = [value2id[v] for v in htc_labels]
        return htc_ids

    def write_data(Xs, spl, dataset):
        print("writing dataset: ", dataset, " ", spl)
        with open(f'./data/{dataset}/hpt_{spl}.json', 'w') as f:
            lines = [json.dumps(X, ensure_ascii=False) + "\n" for X in Xs]
            f.writelines(lines)
        print('done')

    for spl in Xs:
        new_Xs = []
        texts, labels = Xs[spl]['texts'], Xs[spl]['labels']
        for text, label in tqdm(zip(texts, labels), desc=spl):
            new_X = {
                "token": text,
                "label": build_htc_label(label)}
            new_Xs.append(new_X)
        write_data(new_Xs, spl, dataset)
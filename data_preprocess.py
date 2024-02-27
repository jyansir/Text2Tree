# preprocess raw data for PubMed and MIMIC-III
import os
import json
import pandas as pd
import numpy as np
from collections import Counter

# PubMed preprocess utils
def extract_mesh_subids(df, max_level=2):
    """process MeSH IDs"""
    mesh_ids = []
    for mesh in df.meshid.values:
        mesh = eval(mesh)
        each_mesh_ids = []
        for each in mesh:
            each_mesh_ids.extend(list(set(['.'.join(m.split('.')[:max_level]) for m in each])))
        mesh_ids.extend(list(set(each_mesh_ids)))
    id_counter = Counter(mesh_ids)
    mesh_ids = set(mesh_ids)
    return list(mesh_ids), id_counter

def parse_multilabel(meshid, label2id):
    """Convert MeSH IDs into One-hot labels"""
    annote_label = np.zeros(len(label2id), dtype=int)
    for label, id in label2id.items():
        if label in meshid:
            annote_label[id] = 1
    return annote_label.tolist()

# preprocess PubMed and MIMIC-III
def preprocess(data_name, dataset):
    print('preprocessing: ', data_name.upper())
    if data_name == 'pubmed_multilabel':
        df: pd.DataFrame = dataset['train'].to_pandas()
        print('total size: ', len(df))
        # process MeSH IDs
        mesh_ids, id_counter = extract_mesh_subids(df)
        # retain top-100 frequent labels
        common_labels = sorted([label_cnt[0] for label_cnt in id_counter.most_common(100)])
        # labeled by frequency
        label2id = {label: i for i, label in enumerate(common_labels)}
        # avg label number
        avg_label_num = sum(v[1] for v in id_counter.most_common(100)) / 50000
        print('avg label num per sample: ', avg_label_num)
        # convert into one-hot
        annotations = []
        for mesh in df.meshid.values:
            annotations.append(str(parse_multilabel(mesh, label2id)))
        df['ann'] = np.array(annotations) # add `ann` field
        print('one-hot size: ', df['ann'].shape)

        return df, label2id
    elif data_name == 'mimic3-top50':
        # prepare mimic3 data file as:
        # https://github.com/jamesmullenbach/caml-mimic/blob/master/notebooks/dataproc_mimic_III.ipynb
        # to get train, dev, test csv files before executing the scripts
        # please rename them as train(old).csv, dev(old).csv, test(old).csv
        from mimic_utils import codes_50 # import top-50 ICD9 codes
        with open('data/mimic3-top50/icd9label.txt', 'w') as f:
            f.write('\n'.join(codes_50))
        # run provided scripts to convert ICD-9 to ICD-10 (you can manually execute it in cli)
        print('try to convert ICD-9 to ICD-10')
        os.system('python data/mimic3-top50/icd9to10.py data/mimic3-top50/icd9label.txt')
        print('done')
        # read conversion results
        with open('icd9label.txt.out', 'r') as f:
            texts = f.readlines()
        label_dict = {}
        label_text_dict = {}
        for i, text in enumerate(texts):
            if i == 0:
                continue
            icd9, icd10, label_text = text.strip().split('\t')
            if icd10 == 'NA': # drop non-disease codes in top-50 labels
                continue
            if '.' not in icd10:
                icd10 += '.0'
            label_dict[icd9] = icd10
            label_text_dict[icd9] = label_text
        
        new_codes = np.array([label_dict[code] for code in codes_50 if code in label_dict])
        new_code_texts = np.array([label_text_dict[code] for code in codes_50 if code in label_dict])
        sorted_idx = new_codes.argsort()
        new_codes = new_codes[sorted_idx]
        new_code_texts = new_code_texts[sorted_idx]
        # label2id
        code2id = {code: i for i, code in enumerate(new_codes)}
        with open('data/mimic3-top50/label2id.json', 'w') as f:
            json.dump(code2id, f, indent=4)
        # label2text
        code2text = {code: text for code, text in zip(new_codes, new_code_texts)}
        with open('data/mimic3-top50/label2text.json', 'w') as f:
            json.dump(code2text, f, indent=4)
        # update old labels in MIMIC-III datas
        def update_new_label(records):
            for record in records:
                old_labels = record['labels'].split(';')
                new_labels = ';'.join([label_dict[code] for code in old_labels if code in label_dict])
                record['labels'] = new_labels
            return records
        for spl in ['train', 'dev', 'test']:
            with open(f'data/mimic3-top50/{spl}(old).json', 'r') as f:
                records = json.load(f)
            records = update_new_label(records) # update labels
            with open(f'data/mimic3-top50/{spl}.json', 'w') as f:
                json.dump(records, f, indent=4)
    else:
        raise NotImplementedError('Please implement preprocessing pipeline for your own dataset')
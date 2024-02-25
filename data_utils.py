import os
import math
import json
import time
from pathlib import Path
from typing import Callable, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from datasets import load_dataset
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader

from model_utils import MODEL_STATES
from data_preprocess import preprocess

DEFAULT_MODEL_STATES = MODEL_STATES # All experiment training paradigm is in default dataloader behavior
# example: dataloader mode can be customized according to specific model training paradigm
INTRA_MODEL_STATES = [] # ['finetune+intra_mixup'] # only MixUp samples with the same class
INTER_MODEL_STATES = [] # ['pretext+supcon'] # add some pretext tasks for pre-training
DATASET_INFO = {
    'pubmed_multilabel': {
        'task_type': 'multilabel', 'public': True, 
    }, 
    'mimic3-top50': {
        'task_type': 'multilabel', 'public': True, 
    }, 
    'dermatology': {
        'task_type': 'multiclass', 'public': False,
    },
    'gastroenterology': {
        'task_type': 'multiclass', 'public': False,
    },
    'inpatient': {
        'task_type': 'multiclass', 'public': False,
    },
}
DATA = Path('./data')
INPUT_FIELDS = ['input_ids', 'attention_mask'] # used fields of tokenizer output


def proc_htc_label(label_info, label2id, old_id2label, dataset, dummy=False):
    """convert flat text classification labels into HTC label system"""
    def to_one_hot(label):
        first_level, second_level, third_level = label[0], label[:3], label
        one_hot_label = [0] * len(label2id)
        one_hot_label[label2id[first_level]] = 1
        one_hot_label[label2id[second_level]] = 1
        one_hot_label[label2id[third_level]] = 1
        return one_hot_label
    def to_one_hot_dummy(label):
        one_hot_label = [0] * len(label2id)
        one_hot_label[label2id[label]] = 1
        return one_hot_label
    def to_multi_class(label):
        return label2id[label]

    # process pubmed ID or ICD-10 code into HTC label format for HTC baselines
    if dataset in ['gastroenterology', 'dermatology', 'inpatient']:
        label = old_id2label[label_info]
        if not dummy:
            new_label = to_one_hot(label)
        else:
            new_label = to_multi_class(label)
    elif dataset == 'pubmed_multilabel':
        pass
    elif dataset == 'mimic3-top50':
        assert label_info.ndim == 2
    
    return new_label


# Data Utils
def init_data_handler(
    tokenizer,
    data_args,
    splits=['train', 'dev', 'test'], 
    htc_label2id: Optional[dict] = None, # provided HTC label system
    over_sample: bool = False, # whether to perform oversampling
    verbose: bool = True,
    return_raw: bool = False, # whether to return raw texts
):
    print(f"Loading dataset: {data_args.dataset.upper()}")
    if over_sample:
        assert DATASET_INFO[data_args.dataset]['task_type'] == 'multiclass', \
            'Over-sampling only supports multiclass tasks'
        print('Using over-sampling.')
    start = time.time() # data loading time
    Xs = {k: [] for k in splits} # encoded texts
    ys = {k: [] for k in splits} # labels
    counters = {} # label counter
    
    cache_dir = DATA / data_args.dataset
    # used in-house datasets
    if data_args.dataset in ['gastroenterology', 'dermatology', 'inpatient']:
        cache_dir = DATA / f'medical_records/{data_args.dataset}'
        datas = {}
        # load data file
        for split in splits:
            with open(cache_dir / f'{split}.json', 'r') as f:
                datas[split] = json.load(f)
        # load label index file
        with open(cache_dir / 'label2id.json', 'r') as f: 
            label2id = json.load(f)
        old_id2label = None
        if htc_label2id is not None:
            old_id2label = {i: label for label, i in label2id.items()}
            label2id = htc_label2id
        id2label = {i: label for label, i in label2id.items()}
        num_labels = len(label2id)
        # preprocessing for in-house medical records
        def preproc_split(data, spl):
            TEXT_FIELDS = (
                ['SUBJ_COMPLAINT', 'MEDICAL_HISTORY', 'PHY_EX'] # outpatient record fields
                if data_args.dataset in ['gastroenterology', 'dermatology'] 
                else [
                    'chief_complaint', 'medical_history', 'past_medical_history', 
                    'physical_examination', 'auxiliary_examination'] # inpatient record fields
            )
            TARGET_FIELD = 'LABEL' # label field

            labels = []
            texts = []
            for d in data:
                if htc_label2id is None:
                    labels.append(d[TARGET_FIELD])
                else:
                    labels.append(proc_htc_label(d[TARGET_FIELD], htc_label2id, old_id2label, data_args.dataset, data_args.dummy))
                texts.append('\n'.join([d[field] for field in TEXT_FIELDS if d[field] is not None and len(d[field]) > 0]))
            labels = np.array(labels)
            
            # state [resample]: over-sampling (only for multi-class task)
            if over_sample and spl == 'train':
                try:
                    print('try to perform over-sampling')
                    from imblearn.over_sampling import RandomOverSampler
                except:
                    raise ImportError("Please correctly install `imblearn` package \
                        according to: https://github.com/scikit-learn-contrib/imbalanced-learn")
                ros = RandomOverSampler(random_state=0)
                texts = np.array(texts).reshape(-1,1)
                texts, labels = ros.fit_resample(texts, labels)
                texts = texts.reshape(-1).tolist()
            
            if return_raw: # return raw texts
                return {'texts': texts, 'labels': labels}
            
            # tokenization
            encoded = tokenizer.batch_encode_plus(texts, max_length=data_args.max_length, padding=True, truncation=True)
            encoded['labels'] = labels

            return encoded
        
        Xs = {split: preproc_split(datas[split], split) for split in splits} # preprocess each split
    
    # PubMed dataset
    elif data_args.dataset == 'pubmed_multilabel':
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_file = cache_dir / 'train.csv'
        if os.path.exists(cache_file):
            print(f'using cache in: {cache_dir}')
            dfs = {split: pd.read_csv(cache_dir / f'{split}.csv') for split in splits} # read preprocessed data split cache
        else:
            cache_file = cache_dir / 'all.csv'
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file)
            else:
                # load from huggingface repository
                dataset = load_dataset('owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH')
                df, label2id = preprocess('pubmed_multilabel', dataset)
                df.to_csv(cache_file) # save all data
                with open(cache_dir / 'label2id.json', 'w') as f: # save label map
                    json.dump(label2id, f, indent=4)
            assert 'ann' in df.columns, 'Please build fine-grained labels beforehand'
            df_train, df_test = train_test_split(df, random_state=32, test_size=0.2, shuffle=True)
            df_train, df_val = train_test_split(df_train, random_state=32, test_size=0.2)
            dfs = {'train': df_train, 'dev': df_val, 'test': df_test}
            # store cache
            df_train.to_csv(cache_dir / 'train.csv', index=False)
            df_val.to_csv(cache_dir / 'dev.csv', index=False)
            df_test.to_csv(cache_dir / 'test.csv', index=False)

        """new ver. : fine-grained multilabel"""
        with open(cache_dir / 'label2id.json', 'r') as f:
            label2id = json.load(f)
        if htc_label2id is not None:
            label2id = htc_label2id
        num_labels = len(label2id)
        id2label = {i: label for label, i in label2id.items()}

        # single split processing
        def preproc_split(df):
            """new ver. : fine-grained multilabel"""
            labels = np.array([eval(ann) for ann in df['ann'].values])
            if htc_label2id is not None:
                num_samples, num_used_labels = labels.shape
                assert num_used_labels != num_labels # dummy high level labels
                dummy_labels = np.zeros((num_samples, num_labels - num_used_labels), dtype=labels.dtype)
                labels = np.concatenate([dummy_labels, labels], axis=1)
            texts = list(df.abstractText.values)
            if return_raw: # return raw texts
                return {'texts': texts, 'labels': labels}
            # tokenization
            encoded = tokenizer.batch_encode_plus(texts, max_length=data_args.max_length, padding=True, truncation=True)
            encoded['labels'] = labels
            return encoded
        
        Xs = {split: preproc_split(dfs[split]) for split in splits}

    # MIMIC-III
    elif data_args.dataset == 'mimic3-top50':
        # preprocess('mimic3-top50') # please refer to the preprocess function to finish data preparation
        assert os.path.exists(cache_dir / 'train.json'), f'Check preprocessed mimic3 dataset at: {cache_dir}'
        datas = {}
        for split in splits:
            with open(cache_dir / f'{split}.json', 'r') as f:
                datas[split] = json.load(f)
        with open(cache_dir / 'label2id.json', 'r') as f:
            label2id = json.load(f)
        if htc_label2id is not None:
            label2id = htc_label2id
        id2label = {i: label for label, i in label2id.items()}
        num_labels = len(label2id)
        # single split preprocessing
        def preproc_split(data):
            labels = []
            texts = []
            for d in data:
                one_hot_label = np.zeros(num_labels, dtype=int)
                for label in d['labels'].split(';'):
                    if label == '':
                        continue
                    one_hot_label[label2id[label]] = 1
                labels.append(one_hot_label)
                texts.append(d['text'])
            labels = np.stack(labels)
            if return_raw:
                return {'texts': texts, 'labels': labels}
            # tokenization
            encoded = tokenizer.batch_encode_plus(texts, max_length=data_args.max_length, padding=True, truncation=True)
            encoded['labels'] = labels
            return encoded
        
        Xs = {split: preproc_split(datas[split]) for split in splits}

    else:
        raise NotImplementedError("Impelement your private dataset processing pipeline here !")
    
    print(f"done: {time.time() - start} s\n")

    if return_raw:
            return Xs, id2label # return raw texts

    # count labels for each split
    for split in splits:
        ys[split] = Xs[split].pop('labels')
        if ys[split].ndim == 1:
            # multiclass tasks with flat label system
            counters[split] = Counter(ys[split].tolist())
        else:
            # multilabel case: multilabel tasks / HTC label system
            counts = ys[split].sum(0)
            assert len(counts) == num_labels
            counters[split] = Counter({i: counts[i] for i in range(num_labels)})

    # save label counter
    if not over_sample and not os.path.exists(cache_dir / 'label2count.json'):
        print('saving label statistics')
        label2count = {i: int(counters['train'][i]) for i in range(num_labels)}
        with open(cache_dir / 'label2count.json', 'w') as f:
            json.dump(label2count, f, indent=4)
    # save sample number (for dbloss)
    if not over_sample and not os.path.exists(cache_dir / 'data_info.json'):
        print('saving data infos')
        sample_nums = {spl: ys[spl].shape[0] for spl in ['train', 'dev', 'test']}
        with open(cache_dir / 'data_info.json', 'w') as f:
            json.dump(sample_nums, f, indent=4)

    if verbose: # print data infos
        print('tokenizer outputs: ', Xs['train'].keys())
        print("DATASET INFO")
        print(f"[TASK TYPE]: {DATASET_INFO[data_args.dataset]['task_type']}")
        print(f"# CLASSES: {len(id2label)}")
        for split in splits:
            print()
            print(f"# {split} data: {len(ys[split])}")
            print(f">>>>> {split} Label Info <<<<<")
            for id, cnt in sorted(counters[split].items(), key=lambda x: x[1]):
                print(f"# {id2label[id]} -> {cnt}")
    
    return counters, id2label, Xs, ys


class MyLoader:
    """Wrapped DataLoader"""
    def __init__(
        self, 
        Xs, ys, 
        loader_type, # control loader behavior
        batch_size, 
        idx_group_list=None, 
        device=None,
        **kwargs
    ) -> None:
        """
        default: single sentence
        inter: inter-class (give data pair in same class with controlled rate)
        intra: intra-class (give data pair with same class); for intra-class mixup
        TODO: create a medical id guided contrastive learning, idx_group_list add `neighbour` and `non_neigh` for each class
        """
        assert loader_type in ['default', 'inter', 'intra']
        assert all(k in Xs.keys() for k in ['input_ids', 'token_type_ids', 'attention_mask'])

        self.Xs = {k: np.array(v, dtype=np.int64) for k, v in Xs.items()}
        self.ys = ys.astype(np.int64)
        self.dataset = self.ys # provide API for len(MyLoader.datasets)
        self.loader_length = math.ceil(len(self.dataset) / batch_size)
        
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.loader_type = loader_type

        if loader_type == 'inter':
            # use positive pair / negative pair = 1 : 3
            kwargs.setdefault('alpha', 0.75)
            self.alpha = kwargs['alpha'] # control negative sample rate

        # a list of sample index list for each class
        if idx_group_list is None:
            pass
        self.idx_group_list = idx_group_list
        pass
    
    def select_pair(self, ys):
        # choose a sample for each (parallel)
        # return sample index for pair combine
        # update sentence-pair task labels
        if self.loader_type == 'inter':
            """promise positive pairs in Supervised CL"""
            n_neg = math.ceil(len(ys) * self.alpha)
            n_pos = len(ys) - n_neg

            # negative pair
            label_cnt = Counter(ys[:n_neg])
            idx_neg = np.empty((n_neg,), dtype=np.int64)
            for label, cnt in label_cnt.items():
                # select sample with different class
                idx_neg[np.where(ys[:n_neg] == label)] = np.random.choice(
                    self.idx_group_list['neg'][label], size=(cnt,), replace=True)

            # positive pair
            label_cnt = Counter(ys[n_neg:])
            idx_pos = np.empty((n_pos,), dtype=np.int64)
            for label, cnt in label_cnt.items():
                # select sample with same class
                idx_pos[np.where(ys[n_neg:] == label)] = np.random.choice(
                    self.idx_group_list['pos'][label], size=(cnt,), replace=True)
                
            idx = np.concatenate((idx_neg, idx_pos)) # selected negative samples and positive ones
            assert len(idx) == len(ys)
            # ys = np.array([0]*n_neg + [1]*n_pos, dtype=np.int64) # DEBUG: data type ?

        elif self.loader_type == 'intra':
            """for intra-class mixup"""
            # positive pair only
            label_cnt = Counter(ys)
            idx_pos = np.empty((len(ys),), dtype=np.int64)
            for label, cnt in label_cnt.items():
                # select sample with same class
                idx_pos[np.where(ys == label)] = np.random.choice(
                    self.idx_group_list['pos'][label], size=(cnt,), replace=True)
            
            idx = idx_pos

        return idx
    
    def __len__(self):
        return self.loader_length
    
    # iterator
    def __iter__(self):
        """
        return: X, y or None, is_pair
        """
        pos = np.random.permutation(len(self.ys))
        offset = 0
        while offset < len(self.ys):
            idx = pos[offset:min(offset + self.batch_size, len(self.ys))]
            offset += self.batch_size
            # load a batch of single sentence
            X = {k: v[idx] for k, v in self.Xs.items()}
            y = self.ys[idx]
            # IF: need inter or intra loader
            if self.loader_type != 'default':
                # update labels (if needed, for pretext label is not original ys)
                pair_idx = self.select_pair(y) # samples for SupCon or Mixup
                # give sample pair
                X2 = {k: v[pair_idx] for k, v in self.Xs.items()}
                y2 = self.ys[pair_idx]
                # return paired data
                yield {
                    "inputs": (
                        {k: torch.tensor(v, device=self.device) for k, v in X.items()}, 
                        {k: torch.tensor(v, device=self.device) for k, v in X2.items()}
                    ),
                    "labels": (
                        torch.tensor(y, device=self.device),
                        torch.tensor(y2, device=self.device),
                    ),
                    "is_pair": True
                }
            else:
                yield {
                    "inputs": {k: torch.tensor(v, device=self.device) for k, v in X.items()}, 
                    "labels": torch.tensor(y, device=self.device),
                    "is_pair": False
                }



def prepare_loader_for_task(tokenizer, data_args, training_args, loader_type, htc_label2id=None):
    """prepare task dataloaders"""
    splits = ['train', 'dev', 'test']

    # specific processing pipeline
    counters, id2label, encoded, labels = init_data_handler(
        tokenizer, data_args, 
        splits=splits, 
        htc_label2id=htc_label2id,
        over_sample='resample' in loader_type,
    )

    # prepare dataloaders
    dataloaders = {}
    used_fields = INPUT_FIELDS.copy()
    if 'code_rank' not in encoded['train']:
        used_fields.pop(-1)
    for split in splits:
        bs = (
            training_args.per_device_train_batch_size 
            if split == 'train' 
            else training_args.per_device_eval_batch_size
        ) * training_args.n_gpu if torch.cuda.device_count() > 0 else 8
        shuffle = True if split == 'train' else False
        if loader_type in DEFAULT_MODEL_STATES:
            # default: using tensor dataset
            encoded_inputs = [
                torch.tensor(encoded[split][field]) for field in used_fields
            ] + [torch.tensor(labels[split])]
            dataset = TensorDataset(*encoded_inputs)
            dataloaders[split] = DataLoader(dataset, batch_size=bs, shuffle=shuffle)
        else:
            # prepare sample index list grouped by class
            idx_group_list = {'pos': [], 'neg': []}
            label = labels[split]
            for y in range(label.min(), label.max()+1):
                idx_group_list['pos'].append(np.where(label == y)[0])
                idx_group_list['neg'].append(np.where(label != y)[0])

            if loader_type in INTER_MODEL_STATES:
                dl_type = 'inter' if split == 'train' else 'default'
                dataloaders[split] = MyLoader(
                    encoded[split], labels[split],
                    dl_type,
                    bs,
                    idx_group_list=idx_group_list
                )
            elif loader_type in INTRA_MODEL_STATES:
                dl_type = 'intra' if split == 'train' else 'default'
                dataloaders[split] = MyLoader(
                    encoded[split], labels[split],
                    dl_type,
                    bs,
                    idx_group_list=idx_group_list
                )
    return dataloaders, counters, id2label


import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import List

from transformers import AutoTokenizer, BertTokenizer, BertModel, BertConfig

# from graph import GraphEncoder

# label embedding infos: save path, encoder
LABEL_EMBEDDING_CACHE = {
    'pubmed_multilabel': [
        'data/pubmed_multilabel/init_label_embeddings.pth', # label embedding 
        'data/pubmed_multilabel/all_label2text.json', # label text
        'data/pubmed_multilabel/all_label2id.json', # label id
        'bert-base-uncased', # label word encoder
    ],
    'mimic3-top50': [
        'data/mimic3-top50/init_label_embeddings.pth',
        'data/mimic3-top50/all_label2text.json',
        'data/mimic3-top50/all_label2id.json',
        'bert-base-uncased',
    ],
    'dermatology': [
        'data/medical_records/dermatology/init_label_embeddings.pth',
        'data/medical_records/dermatology/all_label2text.json',
        'data/medical_records/dermatology/all_label2id.json',
        'bert-base-chinese',
    ],
    'gastroenterology': [
        'data/medical_records/gastroenterology/init_label_embeddings.pth',
        'data/medical_records/gastroenterology/all_label2text.json',
        'data/medical_records/gastroenterology/all_label2id.json',
        'bert-base-chinese',
    ],
    'inpatient': [
        'data/medical_records/inpatient/init_label_embeddings.pth',
        'data/medical_records/inpatient/all_label2text.json',
        'data/medical_records/inpatient/all_label2id.json',
        'bert-base-chinese',
    ]
}

# label2id = None
# node_id = 1
layer_nodes = [['Root']]
def extract_graph_info(label_list: List[str], sep='.'):
    label_graph = {}
    for label in label_list:
        cur_nodes = label.split(sep)
        cur_nodes = [sep.join(cur_nodes[:i+1]) for i in range(len(cur_nodes))]
        cur_nodes = [cur_nodes[0][0]] + cur_nodes # ['A', 'A01']
        sub_graph = label_graph
        for n in cur_nodes:
            if n not in sub_graph:
                sub_graph[n] = {}
            sub_graph = sub_graph[n]
    return label_graph

def group_labels(label_graph, level=1):
    # global node_id
    if len(label_graph) == 0:
        return
    for label in label_graph:
        # label2id[label] = node_id
        # node_id += 1
        if len(layer_nodes) < level + 1:
            layer_nodes.append([])
        layer_nodes[level].append(label)
    for label in layer_nodes[level]:
        if label in label_graph:
            group_labels(label_graph[label], level+1)

def build_label_map():
    all_labels = []
    for nodes in layer_nodes:
        all_labels.extend(sorted(nodes))
    return {label: i for i, label in enumerate(all_labels)}

# examples of layer nodes
# 1st layer: ['A', 'B', 'C']
# 2nd layer: ['A.a', 'A.b', 'B.a', 'C.a']
# 3rd layer: ['A.a.123', 'A.a.42', 'A.b.231', ...] # Leaf labels for ICD coding
def build_masks(label2id):
    # including Virtual Root Node
    # num_node = sum([len(nodes) for nodes in layer_nodes]) - 1 # exclude Root
    num_node = sum([len(nodes) for nodes in layer_nodes]) # inlcude Root
    num_layer = len(layer_nodes) - 1
    get_parent = lambda child: child[0] if '.' not in child else '.'.join(child.split('.')[:-1])
    attention_masks = []
    for layer_idx in range(1, num_layer+1):
        # p_nodes = layer_nodes[layer_idx-1]
        c_nodes = layer_nodes[layer_idx]
        # attention_mask = np.zeros((num_node+1, num_node+1)) # exclude Root
        attention_mask = np.zeros((num_node, num_node)) # include Root
        # 1: fixed interation, 2: learnable interaction (sibling nodes)
        if layer_idx == 1:
            for child in c_nodes:
                attention_mask[label2id['Root'], label2id['Root']] = 1 # include Root
                attention_mask[label2id[child], label2id['Root']] = 1 # include Root
                for child2 in c_nodes:
                    attention_mask[label2id[child], label2id[child2]] = 1 if child == child2 else 2
        else:
            for child in c_nodes:
                parent = get_parent(child)
                attention_mask[label2id[child], label2id[parent]] = 1
                for child2 in c_nodes:
                    if child == child2:
                        attention_mask[label2id[child], label2id[child2]] = 1
                    else:
                        parent2 = get_parent(child2)
                        if parent == parent2: # sibling nodes
                            attention_mask[label2id[child], label2id[child2]] = 2
        # attention_masks.append(attention_mask[1:,1:]) # exclude Root
        attention_masks.append(attention_mask) # inlcude Root
    return attention_masks, num_node


def extract_mesh_subids(max_level=2, k=100):
    df = pd.read_csv('./data/pubmed_multilabel/all.csv')
    mesh_ids = []
    for mesh in df.meshid.values:
        mesh = eval(mesh)
        each_mesh_ids = []
        for each in mesh:
            each_mesh_ids.extend(list(set(['.'.join(m.split('.')[:max_level]) for m in each])))
        mesh_ids.extend(list(set(each_mesh_ids)))
    id_counter = Counter(mesh_ids)
    mesh_ids = set(mesh_ids)
    common_labels = sorted([label_cnt[0] for label_cnt in id_counter.most_common(k)])
    return list(mesh_ids), id_counter, common_labels


"""Label Embedding Modules"""
import math

import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F

class Attention(nn.Module):
    """Ordinary single-head attention"""
    def __init__(
        self, 
        d: int, 
        # n: int, 
        dropout: float,
        residual_dropout: float = None,
    ) -> None:
        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.connection = nn.Parameter(torch.empty(d, d), requires_grad=True)
        # self.connection = nn.Parameter(torch.empty(n, n), requires_grad=True)
        self.bias = nn.Parameter(torch.ones(1) * 1e-2, requires_grad=True)
        # self.bias = nn.Parameter(torch.empty(1, n), requires_grad=True)
        nn_init.kaiming_uniform_(self.connection, a=math.sqrt(5))
        # nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.residual_dropout = residual_dropout
    
    def calculate_learnable_mask(self, logits, mask):
        soft_mask = torch.sigmoid(logits) # more reasonable design needed
        soft_mask = (soft_mask > 0.5).float() - soft_mask.detach() + soft_mask
        return soft_mask * mask
    
    def forward(self, x, attention_mask, learnable_mask, boundary_mask):
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)

        d_head_key = k.shape[-1]
        attention_logits = q @ k.T / math.sqrt(d_head_key)
        connection_logits = q @ self.connection @ k.T + self.bias
        learnable_mask = self.calculate_learnable_mask(connection_logits, learnable_mask)
        mask = attention_mask + learnable_mask
        mask = -10000 * (1 - mask)

        attention = F.softmax(attention_logits + mask, dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x_residual = attention @ v
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, p=self.residual_dropout, training=self.training)
        # Since only a portion of the labels were used in each attention layer,
        # i.e., labels from adjacent tree levels,
        # we mask out shortcut from unused labels with boundary mask 
        # (only shortcut from labels as keys in this layer is retained)
        x = boundary_mask * x_residual + (1 - boundary_mask) * x
        
        return x


class FixedLabelEmbedding(nn.Module):
    """
    Forzen Label Embedding
    ---

    initialized by encoded label words
    """
    def __init__(self, num_leaf, dataset='pubmed'):
        super().__init__()
        self.num_leaf = num_leaf # number of leaf labels
        self.dataset = dataset
        self.label_embeddings = self.init_embeddings()

    def init_embeddings(self):
        print('fixed label embeddings')
        cache_file = LABEL_EMBEDDING_CACHE[self.dataset][0]
        assert os.path.exists(cache_file)
        print('loading initial embeddings at: ', cache_file)
        init_embeddings = torch.load(cache_file)
        return nn.Parameter(init_embeddings, requires_grad=False)
    
    def forward(self):
        return self.label_embeddings[-self.num_leaf:]

class LabelTreeEmbedding(nn.Module):
    """
    Label embeddings by cascade attention modules
    ---

    Add label tree structure to embed labels,
    this embedding strategy will explicitly regularize 
    labels with common parents (parent labels) sharing similar embeddings

    Args:
        `all_attention_masks`: attention masks of all attention layers
        `num_leaf`: number of leaf labels
        `num_nodes`: number of all labels (including higher-level ones)
        `initialization`: using label word or random initialization for embeddings
    """
    def __init__(self, all_attention_masks, num_leaf, num_nodes=None, dataset='pubmed', dropout=0.1, initialization='label'):
        super().__init__()
        assert initialization in ['label', 'random']
        n_layers = len(all_attention_masks) # number of attention layers
        # fixed attention masks: edge from a parent node to its childs + edge from a node to itself (self-loop)
        attention_masks = torch.stack([torch.from_numpy(mask == 1).float() for mask in all_attention_masks])
        # learnable attention masks: edges among sibling nodes
        learnable_masks = torch.stack([torch.from_numpy(mask == 2).float() for mask in all_attention_masks])
        self.register_buffer('attention_masks', attention_masks)
        self.register_buffer('learnable_masks', learnable_masks)
        self.num_nodes = num_nodes or len(all_attention_masks[0])
        self.num_leaf = num_leaf # number of leaf labels
        self.dataset = dataset

        # initialize in random or with label words
        self.label_embeddings = self.init_embeddings(initialization)
        embed_dim = self.label_embeddings.shape[1]

        # attention qk boundaries
        bool_queries = [[any(m) for m in mask.transpose()] for mask in all_attention_masks]
        bool_keys = [[any(m) for m in mask] for mask in all_attention_masks]
        
        # calculate label boundaries of each attention layer's keys and queries
        # e.g., 1st-layer, query range is from label #0~#17, key range is from label #0~#17
        # e.g., 2st-layer, query range is from label #1~#107, key range is from label #18~#107
        # e.g., 3rd-layer, query range is from label #18~#205, key range is from label #108~#205
        self.boundary_query = []
        self.boundary_key = []
        for i in range(n_layers):
            bq, bk = bool_queries[i], bool_keys[i]
            start_q, start_k = bq.index(True), bk.index(True)
            try:
                end_q = bq[start_q:].index(False) + start_q
                end_k = bk[start_k:].index(False) + start_k
            except:
                end_q = self.num_nodes
                end_k = self.num_nodes
            self.boundary_query.append((start_q, end_q))
            self.boundary_key.append((start_k, end_k))

        self.layers = nn.ModuleList([])
        # we need key boundaries to exclude residual shortcut from uninvolved labels
        boundary_masks = []
        for i in range(n_layers):
            # n_labels = self.boundaries[i][1] - self.boundaries[i][0]
            n_queries = self.boundary_query[i][1] - self.boundary_query[i][0]
            n_keys = self.boundary_key[i][1] - self.boundary_key[i][0]
            boundary_k = (torch.arange(self.num_nodes) >= self.boundary_key[i][0]) & (torch.arange(self.num_nodes) < self.boundary_key[i][1])
            boundary_k = boundary_k.float().unsqueeze(1)
            boundary_masks.append(boundary_k)
            layer = nn.ModuleDict(
                {
                    'attention': Attention(embed_dim, dropout),
                    'norm': nn.LayerNorm(embed_dim),
                }
            )
            self.layers.append(layer)
        boundary_masks = torch.stack(boundary_masks).float()
        self.register_buffer('boundary_masks', boundary_masks)
    
    def init_embeddings(self, initialization):
        """initialize label embeddings"""
        cache_file, label2text_file, label2id_file, model_name = LABEL_EMBEDDING_CACHE[self.dataset]
        if initialization == 'label': # using label words
            print('label embedding initialized from label texts')
            # initialize the label embeddings with label word texts
            # cache_file = 'data/pubmed_multilabel/init_label_embeddings.pth'
            if not os.path.exists(cache_file):
                print('initialize the embedding by Bert Embeddings of label texts')
                with open(label2text_file, 'r') as f:
                    label2text = json.load(f)
                with open(label2id_file, 'r') as f:
                    all_label2id = json.load(f)

                num_labels = len(all_label2id)
                init_embeddings = [None for _ in range(num_labels)]
                tokenizer = BertTokenizer.from_pretrained(model_name)
                bert_embeddings = BertModel.from_pretrained(model_name).embeddings
                for label, i in all_label2id.items():
                    text = label2text[label]
                    if text is None:
                        text = '[MASK]' # for root node
                    input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")['input_ids']
                    init_embedding = bert_embeddings(input_ids=input_ids)[0]
                    # mean of text embeddings
                    init_embeddings[i] = init_embedding.mean(0)
                assert all(e is not None for e in init_embeddings)
                init_embeddings = torch.stack(init_embeddings)
                print('saving initial embeddings at: ', cache_file)
                torch.save(init_embeddings, cache_file)
            else:
                print('loading initial embeddings at: ', cache_file)
                init_embeddings = torch.load(cache_file)
            return nn.Parameter(init_embeddings, requires_grad=True)
        elif initialization == 'random': # randomly initialized
            print('label embedding initialized randomly')
            embed_dim = BertModel.from_pretrained(model_name).embeddings.word_embeddings.weight.shape[1]
            label_embeddings = nn.Parameter(torch.empty(self.num_nodes, embed_dim), requires_grad=True)
            nn_init.kaiming_uniform_(label_embeddings, a=math.sqrt(5))
            return label_embeddings
        else:
            raise NotImplementedError

    def forward(self):
        x = self.label_embeddings
        # Cascade attentions: pass information 
        # 1. from parent labels to childs
        # 2. among siblings
        for layer_idx, layer in enumerate(self.layers):
            x = layer['attention'](
                x, 
                self.attention_masks[layer_idx], 
                self.learnable_masks[layer_idx], 
                self.boundary_masks[layer_idx]
            )
            x = layer['norm'](x)
        return x[-self.num_leaf:] # return leaf label embeddings for final classification

def get_lt_embeddings(dataset='pubmed_multilabel', initialization='random'):
    """Label Tree Embeddings with Cascade Attention Modules"""
    if dataset == 'pubmed_multilabel':
        mesh_ids, id_counter, label_list = extract_mesh_subids()
    elif dataset == 'mimic3-top50':
        with open('data/mimic3-top50/label2id.json', 'r') as f:
            label_list = sorted(list(json.load(f).keys()))
    elif dataset in ['dermatology', 'gastroenterology', 'inpatient']:
        with open(f'data/medical_records/{dataset}/label2id.json', 'r') as f:
            label_list = sorted(list(json.load(f).keys()))
    else:
        raise AssertionError(f'Invalid dataset `{dataset}`')
    label_graph = extract_graph_info(label_list)
    group_labels(label_graph)
    label2id = build_label_map()
    attention_masks, num_node = build_masks(label2id)
    num_leaf_labels = len(label_list)
    if initialization == 'fixed':
        return FixedLabelEmbedding(num_leaf_labels, dataset=dataset)
    else:
        return LabelTreeEmbedding(
            attention_masks, num_leaf_labels, 
            num_nodes=num_node, dataset=dataset, 
            initialization=initialization)
    
# def get_gat_embeddings(bert):
#     """
#     GNN based label embedding
#     ---

#     Reference: https://github.com/wzh9969/HPT/blob/main/train.py
#     """
#     dataset = bert.config.finetuning_task
#     gt = bert.config.task_specific_params['le_init']

#     valuedict_file, slot_file, base_model = LABEL_EMBEDDING_CACHE[dataset]
#     config = BertConfig.from_json_file('data/ge_config.json')
#     data_path = '/'.join(slot_file.split('/')[:-1])
#     label_dict = torch.load(valuedict_file)
#     label_dict = {i: v for i, v in label_dict.items()}
#     setattr(config, 'num_labels', len(label_dict))

#     slot2value = torch.load(slot_file)
#     value2slot = {}
#     num_class = 0
#     for s in slot2value:
#         for v in slot2value[s]:
#             value2slot[v] = s
#             if num_class < v:
#                 num_class = v
#     num_class += 1
#     path_list = [(i, v) for v, i in value2slot.items()]
#     for i in range(num_class):
#         if i not in value2slot:
#             value2slot[i] = -1

#     def get_depth(x):
#         depth = 0
#         while value2slot[x] != -1:
#             depth += 1
#             x = value2slot[x]
#         return depth


#     depth_dict = {i: get_depth(i) for i in range(num_class)}
#     max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
#     depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}

#     for depth in depth2label:
#         for l in depth2label[depth]:
#             path_list.append((num_class + depth, l))

#     depth = len(depth2label)
#     label_dict = torch.load(valuedict_file)
#     tokenizer = AutoTokenizer.from_pretrained(base_model)
#     label_dict = {i: tokenizer.encode(v) for i, v in label_dict.items()}
#     label_emb = []
#     input_embeds = bert.get_input_embeddings()
#     for i in range(len(label_dict)):
#         label_emb.append(
#             input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=bert.device)).mean(dim=0))
#     prefix = input_embeds(torch.tensor([tokenizer.mask_token_id],
#                                         device=bert.device, dtype=torch.long))
#     prompt_embedding = nn.Embedding(depth + 1,
#                                     input_embeds.weight.size(1), 0)
#     label_emb = torch.cat(
#         [torch.stack(label_emb), prompt_embedding.weight[1:, :], prefix], dim=0)
#     label_embedding = GraphEncoder(config, gt, 1, path_list, data_path, n_leaf=bert.config.num_labels)
#     return label_emb, label_embedding


if __name__ == '__main__':
    # Test Label Tree Embedding
    get_lt_embeddings('inpatient')
    mesh_ids, id_counter, label_list = extract_mesh_subids()
    
    label_graph = extract_graph_info(label_list)
    group_labels(label_graph)
    label2id = build_label_map()
    attention_masks, num_node = build_masks(label2id)
    num_leaf_labels = 100

    label_embeddings = LabelTreeEmbedding(attention_masks, num_leaf_labels, num_node)
    # we should use hard distance (node degree difference) more at early stage (because label embeddings are random at the beginning)
    embed_labels = label_embeddings()
    pass
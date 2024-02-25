import json
import numpy as np

from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

from losses import SupConLoss, FocalLoss, SimSurLoss
from balanced_loss_utils import ResampleLoss
from aug_utils import naive_mixup, dis_mixup
from label_embeddings import get_lt_embeddings


MODEL_STATES = [
    'finetune', # ordinary finetune
    'finetune+reweight', 'finetune+resample', # classical methods
    'finetune+focal', 'finetune+dbloss', # weighted loss methods
    'finetune+naivemix', # MixUp
    'finetune+supcon', 'finetune+selfcon', # classical contrastive learning
    # Text2Tree and ablation variants
    'finetune+text2tree', # text2tree
    'finetune+text2tree_DML_only', # only DML with gradient
    'finetune+text2tree_SSL_only', # only SSL
    'finetune+text2tree_gDML', # text2tree with DML gradient
]

def merge_inputs(
    inputs1: Union[torch.Tensor, Dict[str, torch.Tensor]], 
    inputs2: Union[torch.Tensor, Dict[str, torch.Tensor]]
):
    if isinstance(inputs1, dict):
        return {k: torch.cat([inputs1[k], inputs2[k]]) for k in inputs1.keys()}
    return torch.cat([inputs1, inputs2])

def read_class_weights(dataset, return_freq=False):
    """get Label Frequency for DBLoss or Reweighting"""
    # please prepare label count file `label2count.json`
    # or manually count the label frequency here, e.g., {'0': 123, '1': 32}
    count_files = {
        'pubmed_multilabel': 'data/pubmed_multilabel/label2count.json',
        'mimic3-top50': 'data/mimic3-top50/label2count.json',
        'dermatology': 'data/medical_records/dermatology/label2count.json',
        'gastroenterology': 'data/medical_records/gastroenterology/label2count.json',
        'inpatient': 'data/medical_records/inpatient/label2count.json',
    }
    # for DBLoss we need training data size
    num_files = {
        'pubmed_multilabel': 'data/pubmed_multilabel/data_info.json',
        'mimic3-top50': 'data/mimic3-top50/data_info.json',
    }
    assert dataset in count_files
    
    with open(count_files[dataset], 'r') as f:
        label_count = json.load(f)
    label_count = np.array([cnt for cnt in label_count.values()])
    if return_freq: # for DBLoss use
        with open(num_files[dataset], 'r') as f:
            n_train = json.load(f)['train']
        return label_count, n_train
    return 1 - label_count / label_count.sum() # normalized class frequency


class MyModelForBert(BertModel):
    """Wrapped Huggingface Bert Model"""
    # loss function map for contrastive paradigm
    contrast_loss_dict = {
        # classical contrastive loss
        'finetune+supcon': SupConLoss, 'finetune+selfcon': SupConLoss, 
        # Text2Tree and ablation variants
        'finetune+text2tree': SimSurLoss, 
        'finetune+text2tree_DML_only': SimSurLoss, 
        'finetune+text2tree_SSL_only': SimSurLoss, 
        'finetune+text2tree_gDML': SimSurLoss, 
    }

    def __init__(self, *args, **kwargs):
        """
        Model Arguments
        ---
        state: Model learning paradigm
        dout: provided in finetune mode
        task_type(optional): multiclass or multilabel
        """
        super().__init__(*args, **kwargs, add_pooling_layer=False) # manually add BertPooler for MixUp case
        assert 'state' in self.config.task_specific_params, 'Please specify valid model `state` for learning paradigm'
        assert self.config.task_specific_params['state'] in MODEL_STATES, \
            f'Choose from one of the implemented learning paradigms: {MODEL_STATES}'
        assert 'task_type' in self.config.task_specific_params
        self.state = self.config.task_specific_params['state'] # learning paradigm
        
        contrast_params = {} # hyperparams for contrastive paradigms
        # case 1: using contrastive paradigm
        if 'con' in self.state:
            self.config.task_specific_params.setdefault('temperature', 0.1) # temperature
            temperature = self.config.task_specific_params['temperature']
            contrast_params['temperature'] = temperature
        # case 2: finetune with auxiliary loss (finetune loss + lamda * contrastive loss)
        if 'finetune+' in self.state and 'con' in self.state:
            self.config.task_specific_params.setdefault('lamda', 0.1) # auxiliary loss weight
            self.lamda = self.config.task_specific_params['lamda']
        # case 3: using MixUp
        if 'mix' in self.state:
            self.config.task_specific_params.setdefault('alpha', 0.5) # for Beta distribution
            self.alpha = self.config.task_specific_params['alpha']
        # case 4: Label tree embedding for Text2Tree
        if 'text2tree' in self.state:
            downstream_task = self.config.finetuning_task # dataset name
            self.config.task_specific_params.setdefault('le_init', 'random') # label embedding initialization
            le_init = self.config.task_specific_params['le_init']
            if le_init not in ['GAT', 'graphormer']:
                self.label_embeddings = get_lt_embeddings(dataset=downstream_task, initialization=le_init)
            # else:
            #     self.label_emb, self.label_embeddings = self.get_gat_embeddings(dataset=downstream_task, gt=le_init)
                # self.label_embeddings = [self.label_embeddings]
        
        # baseline: reweighting
        if self.state == 'finetune+reweight':
            self.class_weights = read_class_weights(self.config.finetuning_task)
            assert len(self.class_weights) == self.config.num_labels
        
        # baseline: Focal Loss
        if self.state == 'finetune+focal':
            focal_params = {
                'num_labels': self.config.num_labels,
                'task_type': self.config.task_specific_params['task_type']
            }
            if focal_params['task_type'] == 'multilabel':
                self.config.task_specific_params.setdefault('focal_alpha', 0.5)
                self.config.task_specific_params.setdefault('focal_gamma', 2.0)
                focal_params['alpha'] = self.config.task_specific_params['focal_alpha']
                focal_params['gamma'] = self.config.task_specific_params['focal_gamma']
            self.focal_loss_func = FocalLoss(**focal_params)
        
        # baseline: DBLoss
        if self.state == 'finetune+dbloss':
            assert self.config.task_specific_params['task_type'] == 'multilabel' # DBLoss is designed for multilabel
            class_freq, train_num = read_class_weights(self.config.finetuning_task, return_freq=True)
            self.dbloss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                                    focal=dict(focal=True, alpha=0.5, gamma=2),
                                    logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                    map_param=dict(alpha=0.1, beta=10.0, gamma=0.05), 
                                    class_freq=class_freq, train_num=train_num)
        
        # baseline: SelfCon, SupCon, Text2Tree
        if len(contrast_params) > 0: # contrastive loss function
            self.contrast_loss_func = MyModelForBert.contrast_loss_dict[self.state](**contrast_params)
        
        self.task_type = self.config.task_specific_params.get('task_type', None)
        # regular finetune configs
        assert hasattr(self.config, 'num_labels') and self.config.num_labels > 0
        classifier_dropout = self.config.classifier_dropout or self.config.hidden_dropout_prob
        self.pooler = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.Tanh(),
            nn.Dropout(classifier_dropout)
        ) # same as BertPooler
        self.head = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.num_labels = self.config.num_labels
        self.task_type = self.task_type
        # self.task_type = self.task_type or (
        #     'regression' if self.config.task_specific_params['is_regression']
        #     else 'binclass' if self.num_labels == 1
        #     else 'multiclass'
        # )

        # optional projector (refer to classical contrastive learning operation)
        projector_dim = self.config.task_specific_params.get('projector_dim', None)
        if projector_dim is not None:
            self.projector = nn.Linear(self.config.hidden_size, projector_dim)
        else:
            self.projector = nn.Identity()

    def forward(self, inputs, labels=None, label_embeddings=None):
        """Using last hidden states (not pooler out) in MixUp and Contrast"""
        if isinstance(inputs, list):
            inputs = merge_inputs(inputs[0], inputs[1])
            labels = None if not labels else merge_inputs(labels[0], labels[1])
            self.forward(inputs, labels)
        else:
            loss = None
            contrast_loss = None
            
            outputs = super().forward(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[1:] # hidden states from all Bert layers
            all_cls_states = [h[:, 0] for h in hidden_states] # all hidden states at [CLS] position
            last_cls_states = all_cls_states[-1] # [CLS] token state from the last layer

            mixup_cls_states = None
            if 'naivemix' in self.state and self.training:
                # perform ordinary MixUp on [CLS] hidden states
                mixup_cls_states, lambdas, shuffled_ids = naive_mixup(last_cls_states, self.alpha)

            # contrastive paradigm
            if hasattr(self, 'contrast_loss_func') and self.training:
                contrast_inputs = {}

                contrast_inputs['features'] = self.projector(last_cls_states)
                contrast_inputs['labels'] = labels
                if 'selfcon' in self.state: # SelfCon
                    contrast_inputs['selfcon'] = True
                
                if 'text2tree' in self.state: # Text2Tree
                    contrast_inputs['label_embeddings'] = (
                        label_embeddings 
                        if label_embeddings is not None 
                        else self.label_embeddings()
                    ) # use frozen embeddings (if pass `label_embeddings`) or learnable ones
                    contrast_inputs['mixup'] = 'none'
                    contrast_inputs['return_dist'] = 'none'
                    
                    # whether to detach DML
                    if self.state == 'finetune+text2tree': # detach DML process
                        contrast_inputs['return_dist'] = 'detach'
                    elif self.state == 'finetune+text2tree_gDML': # retain gradient of DML
                        contrast_inputs['return_dist'] = 'grad'
                
                if any(k in self.state for k in ['selfcon', 'supcon', 'SSL_only']):
                    # SSL only (Text2Tree w/o DML) or SelfCon / SupCon
                    # assert contrast_inputs['return_dist'] == 'none'
                    assert self.state in ['finetune+selfcon', 'finetune+supcon', 'finetune+text2tree_SSL_only']
                    contrast_loss = self.contrast_loss_func(**contrast_inputs)
                    similarity = None
                else:
                    assert any(k in contrast_inputs['return_dist'] for k in ['detach', 'grad'])
                    if 'DML_only' not in self.state:
                        # Text2Tree
                        assert self.state in ['finetune+text2tree', 'finetune+text2tree_gDML']
                        # SSL, similarity: pairwise similarity score
                        contrast_loss, similarity = self.contrast_loss_func(**contrast_inputs)
                        # DML: dissimilarity based MixUp (label tree MixUp)
                        mixup_cls_states, lambdas = dis_mixup(last_cls_states, similarity)
                        shuffled_ids = None
                    else:
                        # DML only (Text2Tree w/o SSL)
                        assert self.state in ['finetune+text2tree_DML_only']
                        # i.e., contrast_inputs['return_dist'] == 'detach_only' means only using detached DML
                        similarity = self.contrast_loss_func(**contrast_inputs) # only get dissimilarity 
                        contrast_loss, shuffled_ids = None, None                        
                        mixup_cls_states, lambdas = dis_mixup(last_cls_states, similarity) # DML

            # ordinary finetune loss or MixUp loss
            cls_state = last_cls_states if mixup_cls_states is None else mixup_cls_states
            logits = self.head(self.pooler(cls_state))

            if labels is not None: # training, evaluation
                if self.state not in ['finetune+reweight', 'finetune+focal', 'finetune+dbloss']:
                    loss = (
                        self.calculate_finetune_loss(logits, labels) # ordinary finetune loss
                        if mixup_cls_states is None
                        else self.calculate_mixup_loss(lambdas, shuffled_ids, logits, labels) # MixUp or DML loss
                    )
                elif self.state == 'finetune+reweight':
                    loss = self.calculate_weighted_loss(logits, labels) # reweighting loss
                elif self.state == 'finetune+focal':
                    loss = self.focal_loss_func(logits, labels) # focal loss
                else:
                    assert self.state == 'finetune+dbloss'
                    loss = self.dbloss_func(logits, labels) # DBLoss
            
            if contrast_loss is not None and self.training:
                # add auxiliary loss if exists and in training stage
                tot_loss = (1 - self.lamda) * loss + self.lamda * contrast_loss
            else:
                # only finetune loss (labels are accessible)
                # or contrastive loss (if use unsupervised learning, e.g., contrastive self-learning)
                tot_loss = (
                    loss if loss is not None
                    else contrast_loss if contrast_loss is not None
                    else None
                )

            return (tot_loss, logits) if tot_loss is not None else logits # Trainer API
    
    def calculate_finetune_loss(self, logits, labels, reduction='mean'):
        # calculate finetune loss: CE for classification, MES for regression
        if self.task_type == 'multiclass':
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1), reduction=reduction)
        elif self.task_type in ['binclass', 'multilabel']:
            loss = F.binary_cross_entropy_with_logits(logits, labels.type_as(logits), reduction=reduction)
        elif self.task_type == 'regression':
            loss = F.mse_loss(logits.squeeze(), labels.squeeze(), reduction=reduction)
        return loss
        
    def calculate_weighted_loss(self, logits, labels):
        assert hasattr(self, 'class_weights') # reweighting state
        losses = self.calculate_finetune_loss(logits, labels, 'none')
        weights = torch.from_numpy(self.class_weights).to(losses.device)
        if losses.ndim == 2: # classification loss: b, l
            losses = losses * weights.unsqueeze(0)
            losses = losses.sum(1)
        else: # regression loss: b,
            weights = weights[labels.long()] * labels.shape[0]
            losses = losses * weights
        return losses.mean()

    def calculate_mixup_loss(self, mix_masks, shuffled_ids, logits, labels):
        # assert len(mix_masks) == len(shuffled_ids)
        lambdas1, lambdas2 = mix_masks, 1 - mix_masks
        if shuffled_ids is not None:
            # for MixUp loss
            labels_shuffled = labels[shuffled_ids]
        else:
            # for DML loss (b x b MixUp combinations)
            if labels.ndim == 1:
                b = labels.shape[0]
                labels_shuffled = labels.unsqueeze(0).repeat(b, 1).reshape(-1)
                labels = labels.unsqueeze(1).repeat(1, b).reshape(-1)
            elif labels.ndim == 2:
                b, l = labels.shape
                labels_shuffled = labels.unsqueeze(0).repeat(b, 1, 1).reshape(-1, l)
                labels = labels.unsqueeze(1).repeat(1, b, 1).reshape(-1, l)
        if self.task_type == 'regression':
            labels = labels.squeeze()
            mix_labels = lambdas1 * labels + lambdas2 * labels_shuffled
            loss = self.calculate_finetune_loss(logits, mix_labels)
        else:
            loss1 = self.calculate_finetune_loss(logits, labels, 'none')
            loss2 = self.calculate_finetune_loss(logits, labels_shuffled, 'none')
            if '_tmix' in self.state: # DML of Text2Tree
                if loss1.ndim == 2: # multilabel
                    loss = lambdas1.view(-1, 1) * loss1 + lambdas2.view(-1, 1) * loss2
                elif loss1.ndim == 1: # multiclass
                    loss = lambdas1 * loss1 + lambdas2 * loss2
                else:
                    raise AssertionError('Invalid loss dim')
            elif 'naivemix' in self.state: # naive MixUp
                loss = lambdas1 * loss1 + lambdas2 * loss2
            else:
                raise TypeError("Invalid mixup type")
            loss = loss.mean()
        return loss


# class BertGNN(nn.Module):
#     """Wrapped Bert Model with GNN-based Label Embedding (not recommended)"""
#     def __init__(self, bert, gnn, label_emb) -> None:
#         super().__init__()
#         self.bert = bert
#         self.gnn = gnn
#         self.register_buffer('label_emb', label_emb)
    
#     def label_embeddings(self):
#         input_emb = self.bert.get_input_embeddings()
#         return self.gnn(self.label_emb, input_emb)
    

#     def forward(self, input, label=None):
        
#         label_embeddings = self.label_embeddings()
#         # self.gnn.zero_grad(set_to_none=False)
#         # label_embeddings = torch.randn(100,768,device=self.bert.device)
#         return self.bert(input, label, label_embeddings)
    
#     @classmethod
#     def from_pretrained(cls, *args, **kwargs):
#         bert = MyModelForBert.from_pretrained(*args, **kwargs)
#         label_emb, gnn = get_gat_embeddings(bert)
#         return BertGNN(bert, gnn, label_emb)

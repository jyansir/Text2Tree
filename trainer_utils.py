# Trainer
import os
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import classification_report, f1_score, accuracy_score

import torch
import torch.nn as nn

from transformers.trainer import *
from transformers.trainer_callback import TrainerCallback

from data_utils import MyLoader, INPUT_FIELDS


np_sigmoid = lambda x: 1 / (1 + np.exp(-x))
def build_metric(task_type):
    if task_type == 'multiclass':
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            results = classification_report(labels, predictions, output_dict=True)
            # Trainer API will automatically add 'eval_' prefix, e.g., 'eval_macro-f1'
            return {
                'macro-f1': results['macro avg']['f1-score'],
                'micro-f1': results['accuracy'],
                # F1 score per class
                **{f'class{c}-f1': results[c]['f1-score'] for c in results.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']},
            }
    elif task_type == 'multilabel':
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np_sigmoid(logits)

            thresh = 0.5 # default binary threshold
            pred_bools = [pred > thresh for pred in predictions]
            gold_bools = [gold == 1 for gold in labels]
            results = classification_report(gold_bools, pred_bools, output_dict=True)
            # Trainer API will automatically add 'eval_' prefix, e.g., 'eval_macro-f1'
            return {
                'macro-f1': f1_score(gold_bools, pred_bools, average='macro'),
                'micro-f1': f1_score(gold_bools, pred_bools, average='micro'),
                'flat-acc': accuracy_score(gold_bools, pred_bools),
                # F1 score per class
                **{f'class{c}-f1': results[c]['f1-score'] for c in results.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']}
            }
    else:
        return None
    
    return compute_metrics


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()

def input_format(
    inputs: Union[Dict[str, torch.Tensor], List[torch.tensor]],
    device: torch.device = None
):
    if device is None:
        device = 'cuda' if torch.cuda.device_count() else 'cpu'
    if isinstance(inputs, dict):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs.pop('labels', None)
        return inputs, labels
    elif isinstance(inputs, list):
        labels = inputs[-1].to(device) # uniformly place the label at the end
        return {INPUT_FIELDS[i]: data.to(device) for i, data in enumerate(inputs[:-1])}, labels

def has_labels(inputs: Union[Dict[str, torch.Tensor], List[torch.tensor]]):
    if isinstance(inputs, dict):
        return 'labels' in inputs.keys()
    elif isinstance(inputs, list):
        return len(inputs) == len(INPUT_FIELDS) + 1
    
class MyTrainer(Trainer):
    """Wrapped Huggingface Trainer SubClass"""
    def get_train_dataloader(self) -> MyLoader:
        return self.train_dataset
    def get_eval_dataloader(self, eval_dataset=None) -> MyLoader:
        self.eval_dataset = eval_dataset or self.eval_dataset
        return self.eval_dataset
    def get_test_dataloader(self, test_dataset=None) -> MyLoader:
        return test_dataset or self.eval_dataset
    
    def compute_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs=False):
        inputs, labels = input_format(inputs) # parse input format
        outputs = model(inputs, labels) # forward
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        return (loss, outputs) if return_outputs else loss # same as the Trainer API
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        
        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward(retain_graph=True)

        return loss.detach()

    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys=None):
        model_input, labels = input_format(inputs)
        if ignore_keys is None:
            ignore_keys = []

        with torch.no_grad():
            if labels is not None :
                with self.autocast_smart_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    logits = outputs[1]
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                loss = loss.detach()
            else:
                loss = None
                with self.autocast_smart_context_manager():
                    logits = model(model_input) # except for `pretext` with no labels

        if prediction_loss_only:
            return (loss, None, None)
        
        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        
        return (loss, logits, labels)
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and 'label_embeddings' not in n],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and 'label_embeddings' not in n],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if 'label_embeddings' in n],
                    "lr": self.args.learning_rate / 10, # a smaller learning rate for learnable label embeddings
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

# Some Used Callbacks
class SaveLabelEmbeddingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model, **kwargs):
        print('Saving initial label embeddings')
        with torch.no_grad():
            label_embeddings = model.label_embeddings().cpu()
        saved_path = Path(args.output_dir) / 'label_embeddings'
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        torch.save(label_embeddings, saved_path / 'le_init.pth')
    
    def on_epoch_end(self, args, state, control, model, **kwargs):
        print('Saving label embeddings')
        with torch.no_grad():
            label_embeddings = model.label_embeddings().cpu()
        saved_path = Path(args.output_dir) / 'label_embeddings'
        torch.save(label_embeddings, saved_path / f'le_{state.epoch}.pth')

def save_data_info(saved: Dict[str, Any], counters: Dict[str, Counter] = None, split='train', prefix=None):
    prefix = prefix or split # field name, use split as default
    """Save data amount of each class"""
    if counters is not None:
        assert split in counters
        for c, cnt in sorted(counters[split].items(), key=lambda x: x[0]):
            saved[f"{prefix}_samples-{c}"] = cnt
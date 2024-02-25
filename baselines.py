import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import logging
# logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from data_utils import prepare_loader_for_task, DATASET_INFO
from trainer_utils import MyTrainer, build_metric, save_data_info, SaveLabelEmbeddingCallback
from model_utils import MyModelForBert, MODEL_STATES # , BertGNN # uncomment if using GNN-based embeddings
# TODO: add earlystop args for finetune

# DataArguments
@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    max_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    dataset: str = field(
        default='pubmed_multilabel',
        metadata={"help": "Dataset name."},
    )
    data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override dataset path in `data_utils.py` if given."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of examples to this "
                "value if set."
            )
        },
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    # model params
    model_name_or_path: str = field(
        default='bert-base-uncased', metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    # training params
    patience: Optional[int] = field(
        default=None, metadata={"help": "Early stopping"},
    )
    model_state: str = field(
        default='finetune', metadata={"help": "Specific scheme for model training"}
    )
    le_init: str = field(
        default='random', metadata={"help": "initialization of label embedding for HRL"}
    )
    alpha: Optional[float] = field(
        default=None, metadata={"help": "Mixup parameters"}
    )
    focal_alpha: Optional[float] = field(
        default=None, metadata={"help": "Focal loss parameters"}
    )
    focal_gamma: Optional[float] = field(
        default=None, metadata={"help": "Focal loss parameters"}
    )
    temperature: Optional[float] = field(
        default=None,
        metadata={"help": "Temperature of the contrastive learning."},
    )
    lamda: Optional[float] = field(
        default=None,
        metadata={"help": "Ratio of auxiliary loss in finetune."},
    )
    max_gap: Optional[int] = field(
        default=None,
        metadata={"help": "Max gap between two medical codes to be a positive pair."},
    )
    # tokenizer params
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Path to tokenizer, same as model name if not specified"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

# necessary hyperparameters for some learning paradigms
CHECK_MODEL_ARGS = {
    'focal_alpha': (['focal'], 0.5), # Focal Loss
    'focal_gamma': (['focal'], 2.0), # Focal Loss
    'alpha': (['mix'], 0.5), # MixUp
    'temperature': (['selfcon', 'supcon', 'text2tree'], 0.1), # SelfCon, SupCon, Text2Tree
    'lamda': (['finetune+selfcon', 'finetune+supcon', 'finetune+text2tree'], 0.1), # SelfCon, SupCon, Text2Tree
}


def check_args(model_args, data_args, training_args):
    # data args check
    assert data_args.dataset in DATASET_INFO, f'Choose a dataset in {list(DATASET_INFO.keys())}'
    # model args check
    state = model_args.model_state # learning paradigm
    assert state in MODEL_STATES, f'Invalid model status, please choose one of {MODEL_STATES}'
    le_init = model_args.le_init # label embedding initialization
    assert le_init in ['random', 'fixed', 'label', 'GAT', 'graphormer'], f'Invalid label embedding initialization'
    # set necessary args for specific learning paradigm
    for arg_name, (kw_list, default_value) in CHECK_MODEL_ARGS.items():
        if any(kw in state for kw in kw_list):
            print('checking model args: ', arg_name)
            value = getattr(model_args, arg_name)
            if value is None:
                setattr(model_args, arg_name, default_value)
        else:
            setattr(model_args, arg_name, None)
    # training args check
    setattr(training_args, 'lr_scheduler_type', 'constant') # contant lr
    if getattr(model_args, 'patience', None) is None:
        setattr(model_args, 'patience', 10) # default early stop
    call_backs = [EarlyStoppingCallback(model_args.patience)] if model_args.patience > 0 else []
    if 'text2tree' in state: # text2tree save label embedding each epoch
        call_backs.append(SaveLabelEmbeddingCallback)
    return call_backs


def seed_everything(seed=42):
    '''
    Sets the seed of the entire run for REPRODUCIBILITY.
    '''
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    """Prepare experiment arguments"""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # check args for specific learning paradigms
    call_backs = check_args(model_args, data_args, training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint to resume training.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    seed_everything(training_args.seed)

    """process dataset"""
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        # specify the tokenizer or load according to the model
        model_args.tokenizer_name or model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    assert data_args.max_length <= tokenizer.model_max_length # check max length support

    # prepare data loaders, label counter, label index
    dataloaders, counters, id2label = \
        prepare_loader_for_task(tokenizer, data_args, training_args, model_args.model_state)
    
    # label system in FLAT text classification
    # to run HTC baselines, using scripts in the `htc_baselines` folder
    label2id = {v: k for k, v in id2label.items()}

    # prepare hyperparameters
    task_type = DATASET_INFO[data_args.dataset]['task_type'] # multilabel or multiclass
    task_params = dict(
        state=model_args.model_state, 
        task_type=task_type, 
        le_init=model_args.le_init,
    )
    for hyper in CHECK_MODEL_ARGS.keys():
        value = getattr(model_args, hyper, None)
        if value is not None:
            task_params[hyper] = value

    # load config, model, trainer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        id2label=id2label, # saved label infos
        label2id=label2id, # saved label infos
        finetuning_task=data_args.dataset, # dataset name
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision, # model branch
        use_auth_token=True if model_args.use_auth_token else None,
        task_specific_params=task_params, # pass hyperparameters
    )
    if not task_params['le_init'] in ['GAT', 'graphormer']:
        # wrapped BERT model with simple label embedding initialization
        model = MyModelForBert.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision, # model branch
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    else:
        print('using GNN-based label embeddings: ', task_params['le_init'])
        raise AssertionError('Please check GNN related packages are correctly installed, \
            import necessary modules and uncomment the following codes.')
        # we also implement complicated label embedding strategy using GNNs or graphormer
        # such learning process takes more time apparently with insignificant performance change
        # model = BertGNN.from_pretrained(
        #     model_args.model_name_or_path,
        #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #     config=config,
        #     cache_dir=model_args.cache_dir,
        #     revision=model_args.model_revision,
        #     use_auth_token=True if model_args.use_auth_token else None,
        #     ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        # )
        # model.to('cuda')

    # Initialize wrapped Trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer, # will automatically save tokenizer for further load
        train_dataset=dataloaders['train'] if training_args.do_train else None,
        eval_dataset=dataloaders['dev'] if training_args.do_eval else None, # for early stop
        compute_metrics=build_metric(task_type), # metric calculation function
        callbacks=call_backs, # pass callback actions at the end of each epoch or iteration
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        # check resuming training
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        # train model and automatically load the best checkpoint using dev set
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(dataloaders['train'].dataset) # record data volume
        # save_data_info(metrics, counters=counters, split='train') # record data amount of each class, for further analysis
        trainer.save_model()  # also save the tokenizer for easy reloading

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state() # save trainer state

    # Evaluation and Testing
    if training_args.do_eval:
        # The best results on dev set have been recorded in training
        # logger.info("*** Evaluate ***")

        # metrics = trainer.evaluate()
        # metrics["eval_samples"] = len(dataloaders['dev'].dataset)
        # # save_data_info(metrics, counters=counters, split='dev', prefix='eval')

        # trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)

        # testing
        logger.info("*** Testing ***")

        metrics = trainer.evaluate(eval_dataset=dataloaders['test'], metric_key_prefix='test')
        metrics["test_samples"] = len(dataloaders['test'].dataset)
        # save_data_info(metrics, counters=counters, split='test')

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

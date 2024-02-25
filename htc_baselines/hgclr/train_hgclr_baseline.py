# Train HTC baseline: HGCLR
# Reference: https://github.com/wzh9969/contrastive-htc
import sys
import os
import json
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from model.optim import ScheduledOptim, Adam
from tqdm import tqdm
import argparse

sys.path.append('.')
from data_utils import prepare_loader_for_task

sys.path.append('htc_baselines/hgclr')
from eval import evaluate
from model.contrast import ContrastModel
from model.hier import gen_hierarchy
import utils


class Saver:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate.')
parser.add_argument('--model', type=str, default='bert-base-uncased')
parser.add_argument('--dataset', type=str, default='mimic3-top50', choices=['gastroenterology', 'dermatology', 'inpatient', 'mimic3-top50', 'pubmed_multilabel'], help='Dataset.')
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--is_pretraining', default=False, action='store_true')
parser.add_argument('--per_device_train_batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--per_device_eval_batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--early-stop', type=int, default=6, help='Epoch before early stop.')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--name', type=str, required=True, help='A name for different runs.')
parser.add_argument('--update', type=int, default=1, help='Gradient accumulate steps')
parser.add_argument('--warmup', default=0, type=int, help='Warmup steps.')
parser.add_argument('--contrast', default=1, type=int, help='Whether use contrastive model.')
parser.add_argument('--graph', default=1, type=int, help='Whether use graph encoder.')
parser.add_argument('--layer', default=1, type=int, help='Layer of Graphormer.')
parser.add_argument('--multi', default=True, action='store_false', help='Whether the task is multi-label classification.')
parser.add_argument('--lamb', default=1, type=float, help='lambda')
parser.add_argument('--thre', default=0.02, type=float, help='Threshold for keeping tokens. Denote as gamma in the paper.')
parser.add_argument('--tau', default=1, type=float, help='Temperature for contrastive model.')
parser.add_argument('--seed', default=3, type=int, help='Random seed.')
parser.add_argument('--wandb', default=False, action='store_true', help='Use wandb for logging.')
parser.add_argument('--dummy', default=True, action='store_false', help='Use dummy high level label.')


def get_root(path_dict, n):
    ret = []
    while path_dict[n] != n:
        ret.append(n)
        n = path_dict[n]
    ret.append(n)
    return ret

def input_format(
    inputs
    # device: torch.device = None
):
    # if device is None:
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(inputs, dict):
        inputs = {k: v.cuda() for k, v in inputs.items()}
        labels = inputs.pop('labels', None)
        return inputs, labels
    elif isinstance(inputs, list):
        # labels = inputs[-1].cuda()
        return inputs[0].cuda(), inputs[-1].cuda()

inference_task_types = {
    'gastroenterology': 'multiclass', 'dermatology': 'multiclass', 'inpatient': 'multiclass',
    'pubmed_multilabel': 'multilabel', 'mimic3-top50': 'multilabel',
}
inference_label_nums = {
    'gastroenterology': 35, 'dermatology': 59, 'inpatient': 98,
    'pubmed_multilabel': 100, 'mimic3-top50': 33,
}

if __name__ == '__main__':
    args = parser.parse_args()
    device = args.device
    print(args)
    if args.wandb:
        import wandb
        wandb.init(config=args, project='htc')
    utils.seed_torch(args.seed)
    output_dir = f'all_results/{args.dataset}/hgclr'
    # args.name = args.name + '-seed' + str(args.seed) + '-max_len' + str(args.max_length) + '-lr' + str(args.lr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data_path = os.path.join('./data', args.dataset)
    label_dict_file = os.path.join(data_path, 'label_dict.json')
    if not os.path.exists(label_dict_file):
        print('generating label hierarchy files for HGCLR')
        gen_hierarchy(args.model, data_path)
        print('done')
        # with open(os.path.join(data_path, 'all_label2id.json'), 'r') as f:
        #     all_label2id = json.load(f)
        # label_dict = {k: v - 1 for k, v in all_label2id.items() if v != 0}
        # with open(label_dict_file, 'w') as f:
        #     json.dump(label_dict, f, indent=4)
    with open(label_dict_file, 'r') as f:
        label_dict = json.load(f)
    # label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    # label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)

    model = ContrastModel.from_pretrained(args.model, num_labels=num_class,
                                          contrast_loss=args.contrast, graph=args.graph,
                                          layer=args.layer, data_path=data_path, multi_label=args.multi,
                                          lamb=args.lamb, threshold=args.thre, tau=args.tau)
    if args.wandb:
        wandb.watch(model)
    # split = torch.load(os.path.join(data_path, 'split.pt'))
    if args.warmup > 0:
        optimizer = ScheduledOptim(Adam(model.parameters(),
                                        lr=args.lr), args.lr,
                                   n_warmup_steps=args.warmup)
    else:
        optimizer = Adam(model.parameters(),
                         lr=args.lr)

    dataloaders, counters, id2label = \
        prepare_loader_for_task(tokenizer, args, args, 'finetune', htc_label2id=label_dict)
    # label2id = {v: k for k, v in id2label.items()}
    train = dataloaders['train']
    dev = dataloaders['dev']

    # train = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=dataset.collate_fn)
    # dev = DataLoader(dev, batch_size=args.batch, shuffle=False, collate_fn=dataset.collate_fn)
    model.to(device)
    save = Saver(model, optimizer, None, args)
    best_score_macro = 0
    best_score_micro = 0
    early_stop_count = 0
    if not os.path.exists(os.path.join(output_dir, args.name)):
        os.makedirs(os.path.join(output_dir, args.name))
    log_file = open(os.path.join(output_dir, args.name, 'log.txt'), 'w')

    for epoch in range(1000):
        if early_stop_count >= args.early_stop:
            print("Early stop!")
            break
        model.train()
        i = 0
        loss = 0

        # Train
        pbar = tqdm(train)
        for inputs in pbar:
            data, label = input_format(inputs)
            padding_mask = data != tokenizer.pad_token_id
            output = model(data, padding_mask, labels=label, return_dict=True, num_used_labels=inference_label_nums[args.dataset])
            loss /= args.update
            output['loss'].backward()
            loss += output['loss'].item()
            i += 1
            if i % args.update == 0:
                optimizer.step()
                optimizer.zero_grad()
                if args.wandb:
                    wandb.log({'train_loss': loss})
                pbar.set_description('loss:{:.4f}'.format(loss))
                i = 0
                loss = 0
                # torch.cuda.empty_cache()
        pbar.close()

        model.eval()
        pbar = tqdm(dev)
        with torch.no_grad():
            truth = []
            pred = []
            for inputs in pbar:
                data, label = input_format(inputs)
                padding_mask = data != tokenizer.pad_token_id
                output = model(data, padding_mask, labels=label, return_dict=True, num_used_labels=inference_label_nums[args.dataset])
                if label.ndim == 1: # multiclass
                    label = torch.eye(num_class)[label]
                    for l in label:
                        t = []
                        for i in range(l.size(0)):
                            if l[i].item() == 1:
                                t.append(i)
                        truth.append(t)
                else:
                    for l in label:
                        truth.append(l.cpu().numpy())
                for l in output['logits']:
                    pred.append(torch.sigmoid(l).tolist())

        pbar.close()
        scores = evaluate(pred, truth, inference_label_nums[args.dataset], inference_task_types[args.dataset])
        macro_f1 = scores['macro-f1']
        micro_f1 = scores['micro-f1']
        print('macro', macro_f1, 'micro', micro_f1)
        print('macro', macro_f1, 'micro', micro_f1, file=log_file)
        if args.wandb:
            wandb.log({'val_macro': macro_f1, 'val_micro': micro_f1, 'best_macro': best_score_macro,
                       'best_micro': best_score_micro})
        early_stop_count += 1
        if macro_f1 > best_score_macro:
            best_score_macro = macro_f1
            save(macro_f1, best_score_macro, os.path.join(output_dir, args.name, 'checkpoint_best_macro.pt'))
            early_stop_count = 0

        if micro_f1 > best_score_micro:
            best_score_micro = micro_f1
            save(micro_f1, best_score_micro, os.path.join(output_dir, args.name, 'checkpoint_best_micro.pt'))
            early_stop_count = 0
        # save(macro_f1, best_score, os.path.join('checkpoints', args.name, 'checkpoint_{:d}.pt'.format(epoch)))
        # save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_last.pt'))
    log_file.close()

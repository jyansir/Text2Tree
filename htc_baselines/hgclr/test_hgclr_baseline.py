# Test saved HTC baseline: HGCLR
# Reference: https://github.com/wzh9969/contrastive-htc
import sys
import os
import json
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import argparse

sys.path.append('.')
from data_utils import prepare_loader_for_task

sys.path.append('htc_baselines/hgclr')
from eval import evaluate
from model.contrast import ContrastModel

OUTPUT_DIR = 'all_results/hgclr'

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch', type=int, default=32, help='Batch size.')
parser.add_argument('--name', type=str, default="gastroenterology-dummy-53", help='Name of checkpoint. Commonly as DATASET-NAME.')
parser.add_argument('--extra', default='_macro', choices=['_macro', '_micro'], help='An extra string in the name of checkpoint.')
args = parser.parse_args()


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
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, args.name, 'checkpoint_best{}.pt'.format(args.extra)),
                            map_location='cpu')
    batch_size = args.batch
    device = args.device
    extra = args.extra
    args = checkpoint['args'] if checkpoint['args'] is not None else args

    if not hasattr(args, 'graph'):
        args.graph = False
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    data_path = os.path.join('./data', args.dataset)
    with open(os.path.join(data_path, 'label_dict.json'), 'r') as f:
        label_dict = json.load(f)
    # label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    # label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)

    model = ContrastModel.from_pretrained(args.model, num_labels=num_class,
                                          contrast_loss=args.contrast, graph=args.graph,
                                          layer=args.layer, data_path=data_path, multi_label=args.multi,
                                          lamb=args.lamb, threshold=args.thre)
    dataloaders, counters, id2label = \
        prepare_loader_for_task(tokenizer, args, args, 'finetune', htc_label2id=label_dict)
    test = dataloaders['test']
    model.load_state_dict(checkpoint['param'])

    model.to(device)

    truth = []
    pred = []
    index = []
    slot_truth = []
    slot_pred = []

    model.eval()
    pbar = tqdm(test)
    with torch.no_grad():
        for inputs in pbar:
            data, label = input_format(inputs)
            padding_mask = data != tokenizer.pad_token_id
            output = model(data, padding_mask, return_dict=True, )
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

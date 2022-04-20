# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import math
import logging
import torch
import os
import numpy as np
from tqdm import tqdm
import random
import copy
import sys
from collections import Counter, defaultdict
from argparse import ArgumentParser

import sklearn.metrics as metrics

from autorule_model import AutoRule
import utils


device = torch.device('cuda')

parser = ArgumentParser()
# default/required args
parser.add_argument('--data_root', type=str, default='data/', help='Root directory containing datasets (/mnt/exp/data if cluster)')
parser.add_argument('--dataset', type=str, default='sms', help='Name of data dir to use.')
parser.add_argument('--out', type=str, default='OUT', help='Path to output dir.')

# Optional modeling arguments.
parser.add_argument('--num_steps', type=int, default=100, help='Number of training steps per round (default 100).')
parser.add_argument('--n_iter', type=int, default=2, help='Number of self-training rounds (default 2).')
parser.add_argument('--seed', type=int, default=1337, help='Random seed.')
parser.add_argument('--rule_type', type=str, default='ngram-tree', help='Rule generation strategy (default ngram-tree).', choices=['ngram', 'tree-pca', 'ngram-tree'])
parser.add_argument('--soft_labels', type=bool, default=True, help='Whether to use soft or hard labels for distillation, etc (default True).')
parser.add_argument('--distill_threshold', type=int, default=-1, help='Whether to use a threshold on distillation (1) or not (-1, default). TODO make bool.')
parser.add_argument('--teacher_inference', type=bool, default=True,  help='Whether to force teacher inference (default True).')
parser.add_argument('--n_rules', type=int, default=16, help='Number of rules to generate (default 16).')
parser.add_argument('--tree_threshold', type=float, default=0.8, help='Prediction threshold for decision tree rules (default 0.8).')
parser.add_argument('--valid_filter', type=float, default=1, help='The fraction of examples beyond 1.0 will be filtered out. E.g. 1.2, n_rules=15 => build 18 rules then throw out bottom 3. Set to 1 to turn off (this is the default).')
parser.add_argument('--semantic_filter', type=int, default=90, help='Cosine similarity threshold for filtering out rule activations (0 for off). Default is 90.')
parser.add_argument('--train_filter', type=int, default=30, help='Percent of mistakes on training set to throw out (0 for off). Default is 30.')

ARGS = parser.parse_args()




#################################################################
# REORGANIZE ARGUMENTS
#################################################################

fit_params = {
    'n_steps': ARGS.num_steps,
    'optimizer_lr': 5e-5,
    'n_iter': ARGS.n_iter,
    'all_teacher': False,
    'soft_labels': ARGS.soft_labels,
    'distill_threshold': ARGS.distill_threshold,
    'use_unif': True
}
rule_gen_params = {
    'feature_type': ARGS.rule_type,
    'num_features': 1600, 
    'num_rules': ARGS.n_rules,
    'reg_type': 'l2',
    'reg_strength': 1.0,
    'max_df': 0.95,
    'min_df': 4,
    'use_stops': True,
    'dataset': ARGS.dataset,
    'stepwise_inflation_factor': ARGS.valid_filter,
    'semantic_filter_threshold': ARGS.semantic_filter,
    'filter_train_disagree': ARGS.train_filter,
    'pca_tree_threshold': ARGS.tree_threshold
}

if ARGS.dataset in {'cdr', 'youtube', 'sms', 'imdb'}:
    fit_params['metric'] = 'f1_binary'
else:
    fit_params['metric'] = 'f1_macro'



#################################################################
# RUN EXPERIMENT
#################################################################
fit_params['seed'] = ARGS.seed
random.seed(ARGS.seed)
np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)

train_dataset, unlabeled_dataset, valid_data, test_data, full_train = utils.get_train_unlabeled_valid_test(
    ARGS.data_root, ARGS.dataset, prop_labeled=0.05, rule_gen_params=rule_gen_params
)

print('#' * 25)
print('Preparing experiment')
print(f"\t'# train: {len(train_dataset)}")
print(f"\t'# unlabeled: {len(unlabeled_dataset)}")
print(f"\t'# valid: {len(valid_data)}")
print(f"\t'# test: {len(test_data)}")

model = AutoRule(
    batch_size=24, real_batch_size=-1, test_batch_size=48, unsup_prop=0.7,
    teacher_inference=ARGS.teacher_inference, outer_patience=5, rule_embed_size=128,
    backbone='BERT', backbone_model_name='bert-base-cased', optimizer='default')
model.fit(dataset_train=(train_dataset, unlabeled_dataset), dataset_valid=valid_data, device=device, 
    evaluation_step=100, **fit_params)

results = {}
results['valid'], _ = valid_value = model.test(valid_data, fit_params['metric'])
results['test'], preds = test_value = model.test(test_data, fit_params['metric'], write_attns=False)
   



#################################################################
# WRITE RESULTS
#################################################################

out_row = '\t'.join([str(x) for x in [ARGS.dataset, results['valid'], results['test']]])
out_fp = os.path.join(ARGS.out, 'results.tsv')

if not os.path.exists(ARGS.out):
    os.makedirs(ARGS.out)

print('#' * 100)
print('WRITING TO', out_fp)
print('dataset\tvalid\ttest')
print(out_row)
with open(out_fp, 'a') as f:
    f.write(out_row + '\n')

preds_fp =  os.path.join(ARGS.out, 'preds.tsv')
print('WRITING PREDS TO')
with open(preds_fp, 'a') as f:
    for row in preds:
        f.write('\t'.join([str(x) for x in row]) + '\n')

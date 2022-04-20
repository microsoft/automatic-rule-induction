# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from collections import Counter
import sklearn.metrics as metrics
import copy
import random 

from wrench.dataset import load_dataset

import autorule_generator as rules


def coverage(weak_labs, k=1):
    n = 0
    for wl in weak_labs:
        x = [y for y in wl if y != -1]
        if len(x) >= k:
            n += 1
    return float(n) / len(weak_labs)


def pre_f1(weak_labs, labs):
    yhat, y = [], []
    for wl, l in zip(weak_labs, labs):
        x = [y for y in wl if y != -1]
        if len(x) == 0:
            continue
        pred = Counter(x).most_common(1)[0][0]
        yhat.append(pred)
        y.append(l)

    return metrics.precision_score(y, yhat, average='macro'), metrics.f1_score(y, yhat, average='macro')


def get_train_unlabeled_valid_test(dataset_home, data, prop_labeled, rule_gen_params):
    train_dataset, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=False)

    full_train = copy.deepcopy(train_dataset)

    keep_idxs = random.sample(list(range(len(train_dataset))), int(len(train_dataset) * prop_labeled))

    train_dataset, unlabeled_dataset = train_dataset.create_split(idx=keep_idxs)
    unlabeled_dataset.labels = [-1 for _ in unlabeled_dataset.labels]

    train_texts = [ex['text'] for ex in train_dataset.examples]
    train_labs = train_dataset.labels

    # Generate rules and add to dataset
    applier = rules.AutoRuleGenerator(**rule_gen_params)
    applier.train(texts=train_texts, labels=train_labs, 
        unlabeled_texts=[ex['text'] for ex in unlabeled_dataset.examples],
        valid_text=[ex['text'] for ex in valid_data.examples],
        valid_labs=valid_data.labels)

    train_dataset.weak_labels = applier.apply(train_texts, ignore_semantic_filter=True,
        labels=train_labs, filter_train_disagree=rule_gen_params['filter_train_disagree']).tolist()
    train_dataset.n_lf = len(train_dataset.weak_labels[0])

    unlabeled_dataset.weak_labels = applier.apply([ex['text'] for ex in unlabeled_dataset.examples]).tolist()
    unlabeled_dataset.n_lf = len(unlabeled_dataset.weak_labels[0])

    valid_data.weak_labels = applier.apply([ex['text'] for ex in valid_data.examples]).tolist()
    valid_data.n_lf = len(valid_data.weak_labels[0])

    test_data.weak_labels = applier.apply([ex['text'] for ex in test_data.examples]).tolist()
    test_data.n_lf = len(test_data.weak_labels[0])

    # print("RULE QUALITY:")
    # print(f"\tTest coverage: {coverage(test_data.weak_labels, k=1)}")
    # print(f"\tPrecision, F1: {pre_f1(test_data.weak_labels, test_data.labels)}")

    return train_dataset, unlabeled_dataset, valid_data, test_data, full_train


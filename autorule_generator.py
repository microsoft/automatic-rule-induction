# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from collections import Counter
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import random
import os

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer 
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy.stats as stats
from sklearn.metrics import precision_score
import sklearn.metrics as metrics
from sklearn.linear_model._logistic import LogisticRegression

from torch.nn import functional as F
from sentence_transformers import SentenceTransformer


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


def compute_pc(X,npc=1):
    """
    Compute the principal components. 
    X: numpy array [data, features]
    npc: num principal components
    """
    svd = TruncatedSVD(n_components=npc, random_state=0)
    svd.fit(X)
    return svd.components_


def remove_pc(X, npc=1, pc=None):
    """
    Remove the projection on the principal components
    X: numpy array [data, features]
    npc: num principal components
    pc (optional): direction to project and flatten on
    returns: out[i, :] is the data point after removing its projection
    """
    if pc is None:
        pc = compute_pc(X, npc)

    if npc==1:
        out = X - X.dot(pc.transpose()) * pc
    else:
        out = X - X.dot(pc.transpose()).dot(pc)

    return out


def pre(weak_labs, labs):
    yhat, y = [], []
    for wl, l in zip(weak_labs, labs):
        x = [y for y in wl if y != -1]
        if len(x) == 0:
            continue
        pred = Counter(x).most_common(1)[0][0]
        yhat.append(pred)
        y.append(l)

    return metrics.precision_score(y, yhat, average='macro')


class AutoRuleGenerator:

    def __init__(self, 
        feature_type, dataset, num_features=100, 
        num_rules=80, ngram=(1, 3), max_df=0.8, 
        min_df=0.8, use_stops=False, score_type='f1', 
        reg_type='l2', reg_strength=0.5,
        stepwise_inflation_factor=1.0, semantic_filter_threshold=0,
        pca_tree_threshold=0.8, **kwargs):

        self.feature_type = feature_type
        self.num_features = num_features
        self.num_rules = num_rules
        self.ngram = ngram
        self.max_df = max_df
        self.min_df = min_df
        self.use_stops = use_stops
        self.score_type = 'f1'
        self.reg_type = reg_type
        self.reg_strength = reg_strength
        self.trained = False
        self.stepwise_inflation_factor = stepwise_inflation_factor
        self.semantic_filter_threshold = semantic_filter_threshold
        self.dataset = dataset
        self.pca_tree_threshold = pca_tree_threshold
        self.model = SentenceTransformer('all-mpnet-base-v2')


    def train(self, texts, labels, unlabeled_texts=None, valid_text=None, valid_labs=None):
        self.train_texts = texts
        self.train_labels = labels

        if self.feature_type == 'ngram':
            self.featurizer, self.cv = self.build_ngram_featurizer(
                texts=texts,
                labels=labels,
                num_features=self.num_features,
                ngram=self.ngram,
                max_df=self.max_df, min_df=self.min_df, use_stops=self.use_stops
            )
            self.linear_applier(texts, labels, self.featurizer, valid_texts=valid_text, valid_labs=valid_labs)

        elif self.feature_type == 'tree-pca':
            ngram_featurizer, cv = self.build_ngram_featurizer(
                texts=texts,
                labels=labels,
                num_features=self.num_features,
                ngram=self.ngram,
                max_df=self.max_df, min_df=self.min_df, use_stops=self.use_stops)
            X = ngram_featurizer(texts + unlabeled_texts)
            npc = 1
            pc = compute_pc(X, npc=npc)
            Xpc = remove_pc(ngram_featurizer(texts), npc=npc, pc=pc)
            clf = RandomForestClassifier(max_depth=3, random_state=0, 
                n_estimators=int(self.num_rules * self.stepwise_inflation_factor))
            clf.fit(Xpc, labels)
            rule_applier = self.tree_applier_pca(ngram_featurizer, npc, pc, clf, thresh=self.pca_tree_threshold)
            self.applier = rule_applier
            self.trained = True
            rm_idxs = self.stepwise_filter(valid_text, valid_labs, n_to_delete=(len(clf.estimators_) - self.num_rules))
            # returned list is computed with deletions so this wont have indexing errors
            for i in rm_idxs:
                del clf.estimators_[i]

            id2tok = {idx: tok for tok, idx in cv.vocabulary_.items()}
            feature_names = ["'" + id2tok[i]+ "' (+PCA)" for i in range(len(id2tok))]

            self.rule_strs = [
                tree.export_text(clf.estimators_[i], feature_names=feature_names)
                for i in range(len(clf.estimators_))
            ]

            # make new rule applier with updated estimators
            self.applier = self.tree_applier_pca(ngram_featurizer, npc, pc, clf, thresh=self.pca_tree_threshold)

        elif self.feature_type == 'ngram-tree':
            self.featurizer, self.cv = self.build_ngram_featurizer(
                texts=texts,
                labels=labels,
                num_features=self.num_features,
                ngram=self.ngram,
                max_df=self.max_df, min_df=self.min_df, use_stops=self.use_stops
            )
            clf = RandomForestClassifier(max_depth=3, random_state=0, 
                n_estimators=int(self.num_rules * self.stepwise_inflation_factor))
            clf.fit(self.featurizer(texts), labels)
            rule_applier = self.tree_applier(self.featurizer, clf, thresh=self.pca_tree_threshold)
            self.applier = rule_applier
            self.trained = True
            rm_idxs = self.stepwise_filter(valid_text, valid_labs, n_to_delete=(len(clf.estimators_) - self.num_rules))
            # returned list is computed as it deletes, so this wont have indexing errors
            for i in rm_idxs:
                del clf.estimators_[i]

            id2tok = {idx: tok for tok, idx in self.cv.vocabulary_.items()}
            feature_names = [id2tok[i] for i in range(len(id2tok))]

            self.rule_strs = [
                tree.export_text(clf.estimators_[i], feature_names=feature_names)
                for i in range(len(clf.estimators_))
            ]

            # make new rule applier with updated estimators
            self.applier = self.tree_applier(self.featurizer, clf, thresh=self.pca_tree_threshold)

        self.trained = True


    def tree_applier_pca(self, ngram_featurizer, npc, pc, clf, thresh=0.8):
        def rule_applier(txts):
            Xpc = remove_pc(ngram_featurizer(txts), npc=npc, pc=pc)
            out_wl = []
            for row in Xpc:
                wls = []
                for estimator in clf.estimators_:
                    probs = estimator.predict_proba([row])

                    if np.max(probs) > thresh:
                        pred_lab = np.argmax(probs)
                    else:
                        pred_lab = -1
                    wls.append(pred_lab)
                out_wl.append(wls)
            return out_wl

        return rule_applier


    def tree_applier(self, featurizer, clf, thresh=0.8):
        def rule_applier(txts):
            h = featurizer(txts)
            out_wl = []
            for row in h:
                wls = []
                for estimator in clf.estimators_:
                    probs = estimator.predict_proba([row])
                    if np.max(probs) > thresh:
                        pred_lab = np.argmax(probs)
                    else:
                        pred_lab = -1
                    wls.append(pred_lab)
                out_wl.append(wls)
            return out_wl

        return rule_applier


    def filter_rules_semantic(self, side_text, weak_labels, threshold=10):
        assert self.trained
        texts = self.train_texts
        labels = self.train_labels
        train_wl = self.apply(self.train_texts, ignore_semantic_filter=True)
        side_wl = weak_labels

        class2embs = {}
        for cid in set(labels):
            class_texts = [t for t, l in zip(texts, labels) if l == cid]
            emb = self.model.encode(class_texts, normalize_embeddings=True, show_progress_bar=False)
            class2embs[cid] = emb

        all_sims = []
        for ri in tqdm(range(side_wl.shape[1])):
            for pred_cls in set(side_wl[:, ri].tolist()) - {-1}:
                idxs = [i for i, x in enumerate(side_wl[:, ri]) if x == pred_cls]
                firing_texts = [side_text[i] for i in idxs]
                emb = self.model.encode(firing_texts, normalize_embeddings=True, show_progress_bar=False)

                if (len(emb) == 0) or (pred_cls not in class2embs) or (len(class2embs[pred_cls]) == 0):
                    continue

                # maxpool looks like it works best
                sims = np.max(np.dot(emb, class2embs[pred_cls].T), axis=1)
                # sims = np.max(np.dot(emb, rule2embs[ri].T), axis=1)
                # sims = np.dot(emb, np.mean(class2embs[tgt_cls].T, axis=1))

                all_sims += list(zip(sims.tolist(), idxs, [ri] * len(idxs)))

        remove_sims = sorted(all_sims)[:int(len(all_sims) * (threshold * 0.01))]
        for _, i, r in remove_sims:
            side_wl[i, r] = -1

        return side_wl


    def linear_applier(self, texts, labels, featurizer, valid_texts=None, valid_labs=None):
        model = LogisticRegression(
            penalty=self.reg_type, C=self.reg_strength, fit_intercept=False, solver='liblinear')
        X = featurizer(texts)
        model.fit(X, labels)

        n_classes = model.coef_.shape[0]

        name_rulefn_score = []

        if n_classes == 1:
            for idx, weight in enumerate(model.coef_[0]):
                if weight == 0: 
                    continue
                if weight > 0:
                    name_rulefn_score.append( (idx, 1, weight) )
                else:
                    name_rulefn_score.append( (idx, 0, weight) )

        else:
            for class_id in range(n_classes):
                for idx, weight in enumerate(model.coef_[class_id]):
                    if weight <= 0:
                        continue

                    name_rulefn_score.append( (idx, class_id, weight) )

        name_rulefn_score = sorted(name_rulefn_score, key=lambda x: abs(x[-1]), reverse=True)

        name_rulefn_score = name_rulefn_score[:int(self.num_rules * self.stepwise_inflation_factor)]


        self.name_rulefn_score = name_rulefn_score
        self.trained = True

        if self.stepwise_inflation_factor > 1.0:
            rm_idxs = self.stepwise_filter(valid_texts, valid_labs, n_to_delete=(len(name_rulefn_score) - self.num_rules))
            for idx in rm_idxs:
                del self.name_rulefn_score[idx]

        self.rule_strs = []
        idx2tok = {idx: ngram for ngram, idx in self.cv.vocabulary_.items()}
        for i, lab, _ in self.name_rulefn_score:
            self.rule_strs.append(f'{idx2tok[i]} => {lab}')


    def stepwise_filter(self, valid_texts, valid_labs, n_to_delete):
        """ you are good to use the deleted indices in order """
        assert self.trained

        valid_wl = self.apply(valid_texts, ignore_semantic_filter=True)
        rm_idxs = []

        for round in range(n_to_delete):
            i = 0
            best_idx = -1
            best_delta = 0

            # TODO -- possibly f1 score?
            base_precision = pre(valid_wl, valid_labs)

            while i < valid_wl.shape[1]:
                tmp_wl = np.delete(valid_wl, i, axis=1)
                precision = pre(tmp_wl, valid_labs)
                if (precision - base_precision) > best_delta:
                    best_delta = precision - base_precision
                    best_idx = i
                i += 1

            if best_idx > 0:
                valid_wl = np.delete(valid_wl, best_idx, axis=1)
                rm_idxs.append(best_idx)

        return rm_idxs

    def report_rules(self):
        assert self.trained
        if self.feature_type in {'ngram', 'ngram-tree', 'ngram-pca', 'tree-pca'}:
            return self.rule_strs
        else:
            raise NotImplementedError


    def apply(self, texts, labels=None, ignore_semantic_filter=False, filter_train_disagree=0):
        assert self.trained, 'Must TRAIN rule generator before application'

        if self.feature_type == 'ngram':
            X = self.featurizer(texts)
            weak_labels = []
            for row, text in zip(X, texts):
                weak_labs = []
                for i, (idx, label, _) in enumerate(self.name_rulefn_score):
                    if row[idx] == 0:
                        weak_labs.append(-1)
                    else:
                        weak_labs.append(label)
                weak_labels.append(weak_labs)

            weak_labels = np.array(weak_labels)

        elif self.feature_type in {'ngram-pca', 'ngram-tree'}:
            weak_labels = self.applier(texts)
            weak_labels = np.array(weak_labels)

        if (not ignore_semantic_filter) and self.semantic_filter_threshold > 0:
            weak_labels = self.filter_rules_semantic(texts, weak_labels, self.semantic_filter_threshold)

        if labels is not None and filter_train_disagree > 0:
            misfires = []
            for i in range(len(labels)):
                firing_rules = [(i, j) for j in range(len(weak_labels[i])) if 
                    (weak_labels[i, j] != -1 and weak_labels[i, j] != labels[i])
                ]
                misfires += firing_rules

            random.shuffle(misfires)
            take_idx = int(len(misfires) * (filter_train_disagree * 0.01))
            for x, y in misfires[:take_idx]:
                weak_labels[x, y] = -1
        return weak_labels



    def build_ngram_featurizer(self, texts, labels, num_features, ngram, max_df, min_df, use_stops):
        self.tokenizer = LemmaTokenizer()
        
        cv = CountVectorizer(
            tokenizer=self.tokenizer,
            ngram_range=ngram,
            max_features=num_features,
            lowercase=False,
            binary=True,
            stop_words='english' if use_stops else None,
            max_df=max_df,
            min_df=min_df
        )
        cv.fit(texts)

        def featurize(texts):
            corpus_counts = cv.transform(texts)
            corpus_counts = corpus_counts.toarray()
            return corpus_counts

        return featurize, cv


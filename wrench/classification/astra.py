import copy
import logging
from typing import Any, Optional, Union, Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import trange
from transformers import AutoTokenizer

from ..backbone import BackBone
from ..basemodel import BaseTorchClassModel
from ..config import Config
from ..dataset import BaseDataset
from ..utils import cross_entropy_with_probs

import os

logger = logging.getLogger(__name__)

ABSTAIN = -1


class RuleAttentionTeacherNetwork(BackBone):
    def __init__(self, rule_embed_size, n_rules, n_class, hidden_size, dropout, use_unif=True):
        super(RuleAttentionTeacherNetwork, self).__init__(n_class=n_class)
        self.use_unif = use_unif
        self.rule_embedding = nn.Sequential(
            nn.Linear(rule_embed_size, n_rules),
            nn.Sigmoid(),
        )
        self.student_embedding = nn.Sequential(
            nn.Linear(rule_embed_size, 1),
            nn.Sigmoid(),
        )
        self.fcs = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, rule_embed_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, batch, features, proba, return_attns=False):
        device = self.get_device()
        weak_labels = batch['weak_labels'].long().to(device)
        mask = weak_labels != ABSTAIN
        weak_labels_one_hot = F.one_hot(weak_labels * mask, num_classes=self.n_class)

        fc_h = self.fcs(features)
        rule_attention = self.rule_embedding(fc_h) * mask
        student_attention = self.student_embedding(fc_h)
        uniform_weight = torch.sum(mask, dim=1, keepdim=True) + 1 - student_attention - torch.sum(rule_attention, dim=1, keepdim=True)

        weighted_rule = rule_attention.unsqueeze(2) * weak_labels_one_hot
        weighted_proba = student_attention * torch.softmax(proba, dim=1)
        weighted_uniform = uniform_weight / self.n_class
        if self.use_unif:
            weighted_sum = torch.sum(weighted_rule, dim=1) + weighted_proba + weighted_uniform
        else:
            weighted_sum = torch.sum(weighted_rule, dim=1) + weighted_proba
        prediction = weighted_sum / torch.sum(weighted_sum, dim=1, keepdim=True)


        if return_attns:
            out_attns = []
            wl_cpu = weak_labels.cpu()
            attns_cpu = rule_attention.detach().cpu()
            for bi in range(len(wl_cpu)):
                tmp = []
                for ri in range(len(wl_cpu[0])):
                    if wl_cpu[bi][ri] != -1:
                        tmp.append((ri, wl_cpu[bi][ri].item(), attns_cpu[bi][ri].item()))
                out_attns.append(tmp[:])

            return prediction, out_attns

        return prediction


class AstraModel(BackBone):
    def __init__(self, rule_embed_size, dropout, n_rules, n_class, backbone, train_teacher=False, use_unif=True):
        super(AstraModel, self).__init__(n_class=n_class)
        self.backbone = backbone
        self.ran = RuleAttentionTeacherNetwork(rule_embed_size, n_rules, n_class, backbone.hidden_size, dropout, use_unif=use_unif)
        self.train_teacher = train_teacher

    def forward_teacher(self, batch, features=None, proba=None, train_backbone=False, return_attns=False):
        if features is None or proba is None:
            if train_backbone or self.train_teacher: 
                proba, features = self.backbone(batch, return_features=True)
            else:
                with torch.no_grad():
                    proba, features = self.backbone(batch, return_features=True)
        return self.ran(batch, features, proba, return_attns=return_attns)

    def forward(self, batch, return_features=False):
        return self.backbone(batch, return_features)


def update_state_dict(model, state_dict: dict, mode: str):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(mode):
            new_state_dict[k[len(mode) + 1:]] = v
    getattr(model, mode).load_state_dict(new_state_dict)




class Astra(BaseTorchClassModel):
    def __init__(self,
                 n_iter: Optional[int] = 25,
                 outer_patience: Optional[int] = 3,
                 rule_embed_size: Optional[int] = 100,
                 dropout: Optional[float] = 0.3,
                 unsup_prop=1.0,

                 batch_size: Optional[int] = 16,
                 real_batch_size: Optional[int] = 16,
                 test_batch_size: Optional[int] = 16,
                 n_steps: Optional[int] = 10000,
                 grad_norm: Optional[float] = -1,
                 use_lr_scheduler: Optional[bool] = False,
                 binary_mode: Optional[bool] = False,
                 teacher_inference = False,
                 **kwargs: Any
                 ):
        super().__init__()
        self.hyperparas = {
            'n_iter'          : n_iter,
            'outer_patience'  : outer_patience,
            'rule_embed_size' : rule_embed_size,
            'dropout'         : dropout,

            'batch_size'      : batch_size,
            'real_batch_size' : real_batch_size,
            'test_batch_size' : test_batch_size,
            'n_steps'         : n_steps,
            'grad_norm'       : grad_norm,
            'use_lr_scheduler': use_lr_scheduler,
            'binary_mode'     : binary_mode,
        }
        self.model: Optional[AstraModel] = None
        self.config = Config(
            self.hyperparas,
            use_optimizer=True,
            use_lr_scheduler=use_lr_scheduler,
            use_backbone=True,
            use_label_model=False,
            **kwargs
        )
        self.teacher_inference = teacher_inference
        self.is_bert = self.config.backbone_config['name'] == 'BERT'
        if self.is_bert:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_config['paras']['model_name'])
        self.unsup_prop=unsup_prop

    def fit(self,
            dataset_train: None,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,
            pretrained_model: str = None,
            valid_mode: Optional[str] = 'student',
            soft_labels: Optional[bool] = False,
            evaluation_step: Optional[int] = 100,
            metric: Optional[Union[str, Callable]] = 'acc',
            direction: Optional[str] = 'auto',
            patience: Optional[int] = 20,
            tolerance: Optional[float] = -1.0,
            device: Optional[torch.device] = None,
            verbose: Optional[bool] = True,
            stu_sup = None,
            stu_unsup = None,
            tea_sup = None,
            tea_unsup = None,
            all_teacher = False,
            distill_threshold = -1,
            use_unif = True,
            **kwargs: Any):

        if not verbose:
            logger.setLevel(logging.ERROR)

        config = self.config.update(**kwargs)

        config.backbone_config['paras']['max_tokens'] = 256  # TODO hack to get this working

        hyperparas = self.config.hyperparas
        logger.info(config)

        dataset_train, dataset_unlabeled = dataset_train

        n_rules = dataset_train.n_lf
        n_class = dataset_train.n_class
        # labeled_dataset, unlabeled_dataset = dataset_train.create_split(labeled_data_idx)
        labeled_dataset = dataset_train
        unlabeled_dataset = dataset_unlabeled

        n_steps = hyperparas['n_steps']
        # TODO HACK
        if any([stu_sup, stu_unsup, tea_sup, tea_unsup]) is None:
            step_balance = {
                'student-supervised': n_steps,
                'student-unsupervised': n_steps,
                'teacher-supervised': n_steps,
                'teacher-unsupervised': n_steps,
            }
        else:
            step_balance = {
                'student-supervised': stu_sup,
                'student-unsupervised': stu_unsup,
                'teacher-supervised': tea_sup, 
                'teacher-unsupervised': tea_unsup,
            }

        if all_teacher:
            step_balance['student-supervised'] = 0
            step_balance['student-unsupervised'] = 0

        if hyperparas['real_batch_size'] == -1 or hyperparas['batch_size'] < hyperparas['real_batch_size'] or not self.is_bert:
            hyperparas['real_batch_size'] = hyperparas['batch_size']
        accum_steps = hyperparas['batch_size'] // hyperparas['real_batch_size']

        assert config.backbone_config['name'] != 'LogReg'
        backbone = self._init_model(
            dataset=dataset_train,
            n_class=dataset_train.n_class,
            config=config,
            is_bert=self.is_bert
        )
        
        model = AstraModel(
            rule_embed_size=hyperparas['rule_embed_size'],
            dropout=hyperparas['dropout'],
            n_rules=n_rules,
            n_class=n_class,
            backbone=backbone,
            train_teacher=all_teacher,
            use_unif=use_unif
        )
        self.model = model.to(device)

        labeled_dataloader = self._init_train_dataloader(
            labeled_dataset,
            n_steps=n_steps,
            config=config,
            return_weak_labels=True,
            return_labels=True,
            # max_tokens=256
        )

        unlabeled_dataloader = self._init_train_dataloader(
            unlabeled_dataset,
            n_steps=n_steps,
            config=config,
            return_weak_labels=True, 
            # max_tokens=256
        )

        valid_flag = self._init_valid_step(
            dataset_valid,
            y_valid,
            metric,
            direction,
            patience,
            tolerance,
            return_weak_labels=True,
        )
        STEPS = 0

        history = {}



        if pretrained_model is not None:
            logger.info(f'loading pretrained model, so skip pretraining stage!')
            self.model.backbone.load_state_dict(pretrained_model)
        else:
            optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model.backbone, config)

            history_pretrain = {}
            last_step_log = {}
            with trange(n_steps, desc="[TRAIN] ASTRA pretrain student", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for batch in labeled_dataloader:
                    predict_l = model(batch)
                    loss = cross_entropy_with_probs(predict_l, batch['labels'].to(device))
                    loss.backward()
                    cnt += 1
                    if cnt % accum_steps == 0:
                        if hyperparas['grad_norm'] > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), hyperparas['grad_norm'])
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()
                        step += 1
                        STEPS += 1
                        if valid_flag and STEPS % evaluation_step == 0:
                            STEPS = 0
                            metric_value, early_stop_flag, info = self._valid_step(step, mode='student')
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history_pretrain[step] = {
                                'loss'              : loss.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history_pretrain[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        if step >= n_steps:
                            break

            if valid_flag:
                update_state_dict(self.model, self.best_model, 'backbone')

            history['pretrain'] = history_pretrain

        history_train = {}
        last_step_log = {}
        n_iter = hyperparas['n_iter']

        if valid_flag:
            outer_patience = hyperparas['outer_patience']
            outer_no_improve_cnt = 0
            outer_best_model = None
            if self.direction == 'maximize':
                outer_best_metric_value = -np.inf
            else:
                outer_best_metric_value = np.inf
        for i in range(n_iter):
            if valid_flag:
                self._reset_valid()
                self._valid_step(-1, mode='teacher')

            pseudo_probas_u, features_u = self.collect_pseudodataset_student(unlabeled_dataset)
            pseudo_probas_l, features_l = self.collect_pseudodataset_student(labeled_dataset)

            optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model.ran, config)

            history_train_teacher = {}
            with trange(int(n_steps * self.unsup_prop), desc=f"[TRAIN@{i}] ASTRA-teacher-unsup", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for unlabeled_batch in unlabeled_dataloader:
                    # if step_balance['teacher-unsupervised'] == 0: break                
                    idx_u = unlabeled_batch['ids'].long().to(device)
                    predict_u = model.forward_teacher(unlabeled_batch, features_u[idx_u], pseudo_probas_u[idx_u])
                    loss = - torch.mean(torch.sum(predict_u * torch.log(predict_u), dim=-1))
                    loss.backward()
                    cnt += 1
                    if cnt % accum_steps == 0:
                        # if step_balance['teacher-unsupervised'] == 0: break                
                        if hyperparas['grad_norm'] > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), hyperparas['grad_norm'])
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()
                        step += 1
                        STEPS += 1
                        if valid_flag and STEPS % evaluation_step == 0:
                            STEPS = 0
                            metric_value, early_stop_flag, info = self._valid_step(step, mode='teacher')
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history_train_teacher[step] = {
                                'loss'              : loss.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history_train_teacher[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        # if step >= step_balance['teacher-unsupervised']:
                        #     break

            if valid_flag:
                update_state_dict(self.model, self.best_model, 'ran')
                self._reset_valid()
                self._valid_step(-1, mode='teacher')

            optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model.ran, config)

            history_finetune_teacher = {}
            with trange(n_steps, desc=f"[FINETUNE@{i}] ASTRA-teacher-sup", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for label_batch in labeled_dataloader:
                    # if step_balance['teacher-supervised'] == 0: break                
                    idx_l = label_batch['ids'].long().to(device)
                    predict_l = model.forward_teacher(label_batch, features_l[idx_l], pseudo_probas_l[idx_l])
                    loss = cross_entropy_with_probs(predict_l, label_batch['labels'].to(device))
                    loss.backward()
                    cnt += 1
                    if cnt % accum_steps == 0:
                        if hyperparas['grad_norm'] > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), hyperparas['grad_norm'])
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()
                        step += 1
                        STEPS += 1
                        if valid_flag and STEPS % evaluation_step == 0:
                            STEPS = 0
                            metric_value, early_stop_flag, info = self._valid_step(step, mode='teacher')
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history_finetune_teacher[step] = {
                                'loss'              : loss.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history_finetune_teacher[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        # if step >= step_balance['teacher-supervised']:
                        #     break

            if valid_flag:
                update_state_dict(self.model, self.best_model, 'ran')
                self._reset_valid()
                self._valid_step(-1, mode='student')

            optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model.backbone, config)

            pseudo_probas_u = self.collect_pseudodataset_teacher(unlabeled_dataset)

            if distill_threshold > 0:
                loss_weights = (torch.max(torch.nn.functional.softmax(pseudo_probas_u, dim=-1), dim=-1).values > 0.98).float()
            else:
                loss_weights = torch.tensor([1.] * len(pseudo_probas_u), dtype=float).to(device)

            if not soft_labels:
                pseudo_probas_u = torch.argmax(pseudo_probas_u, dim=-1)

            history_train_student = {}
            with trange(int(n_steps * self.unsup_prop), desc=f"[TRAIN@{i}] ASTRA-student-unsup", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for unlabeled_batch in unlabeled_dataloader:
                    # if step_balance['student-unsupervised'] == 0: break                                
                    idx_u = unlabeled_batch['ids'].long().to(device)
                    # print(unlabeled_batch['input_ids'].shape)
                    predict_u = model(unlabeled_batch)
                    loss = cross_entropy_with_probs(predict_u, pseudo_probas_u[idx_u],
                        tok_weight=loss_weights[idx_u])
                    loss.backward()
                    cnt += 1
                    if cnt % accum_steps == 0:
                        if hyperparas['grad_norm'] > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), hyperparas['grad_norm'])
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()
                        step += 1
                        STEPS += 1
                        if valid_flag and STEPS % evaluation_step == 0:
                            STEPS = 0
                            metric_value, early_stop_flag, info = self._valid_step(step, mode='student')
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history_train_student[step] = {
                                'loss'              : loss.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history_train_student[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        # if step >= step_balance['student-unsupervised']:
                        #     break

            if valid_flag:
                update_state_dict(self.model, self.best_model, 'backbone')
                self._reset_valid()
                self._valid_step(-1, mode='student')

            optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model.backbone, config)

            history_finetune_student = {}
            with trange(n_steps, desc=f"[FINETUNE@{i}] ASTRA-student-sup", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for label_batch in labeled_dataloader:
                    # if step_balance['student-supervised'] == 0: break                
                    predict_l = model(label_batch)
                    loss = cross_entropy_with_probs(predict_l, label_batch['labels'].to(device))
                    loss.backward()
                    cnt += 1
                    if cnt % accum_steps == 0:
                        if hyperparas['grad_norm'] > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), hyperparas['grad_norm'])
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()
                        step += 1
                        STEPS += 1
                        if valid_flag and STEPS % evaluation_step == 0:
                            STEPS = 0
                            metric_value, early_stop_flag, info = self._valid_step(step, mode='student')
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history_finetune_student[step] = {
                                'loss'              : loss.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history_finetune_student[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        # if step >= step_balance['student-supervised']:
                        #     break

            if valid_flag:
                update_state_dict(self.model, self.best_model, 'backbone')
                metric_value, _, _ = self._valid_step(i, mode=valid_mode)
                if (self.direction == 'maximize' and metric_value > outer_best_metric_value) or \
                        (self.direction == 'minimize' and metric_value < outer_best_metric_value):
                    outer_best_metric_value = metric_value
                    outer_no_improve_cnt = 0
                    outer_best_model = copy.deepcopy(self.model.state_dict())
                else:
                    outer_no_improve_cnt += 1
                    if outer_patience > 0 and outer_no_improve_cnt >= outer_patience:
                        logger.info(f'[INFO] early stop outer loop @ iteration {i}')
                        break

            history_train[i] = {
                'train_teacher'   : history_train_teacher,
                'finetune_teacher': history_train_teacher,
                'train_student'   : history_train_teacher,
                'finetune_student': history_train_teacher,
            }

        self._finalize()
        if valid_flag and outer_best_model is not None:
            self.model.load_state_dict(outer_best_model)

        history['train'] = history_train
        return history

    @torch.no_grad()
    def collect_pseudodataset_student(self, dataset):
        model = self.model
        model.eval()
        if isinstance(dataset, BaseDataset):
            valid_dataloader = self._init_valid_dataloader(
                dataset,
            )
        else:
            valid_dataloader = dataset
        features, probas = [], []
        for batch in valid_dataloader:
            output, feature = model(batch, return_features=True)
            proba = F.softmax(output, dim=-1)
            probas.append(proba)
            features.append(feature)

        return torch.vstack(probas), torch.vstack(features)

    @torch.no_grad()
    def collect_pseudodataset_teacher(self, dataset):
        model = self.model
        model.eval()
        if isinstance(dataset, BaseDataset):
            valid_dataloader = self._init_valid_dataloader(
                dataset,
                return_weak_labels=True,
            )
        else:
            valid_dataloader = dataset
        probas = []
        for batch in valid_dataloader:
            proba = model.forward_teacher(batch)
            probas.append(proba)

        return torch.vstack(probas)

    @torch.no_grad()
    def predict_proba(self, dataset: Union[BaseDataset, DataLoader], mode: Optional[str] = 'student',
                      device: Optional[torch.device] = None, write_attns=False, **kwargs: Any):
        assert mode in ['teacher', 'student'], f'mode: {mode} not support!'
        if self.teacher_inference:
            mode = 'teacher'

        if device is not None:
            model = self.model.to(device)
        else:
            model = self.model
        model.eval()
        if isinstance(dataset, BaseDataset):
            valid_dataloader = self._init_valid_dataloader(
                dataset,
                return_weak_labels=mode == 'teacher',
            )
        else:
            valid_dataloader = dataset
        probas = []
        attns_tot = []
        for batch in valid_dataloader:
            if mode == 'teacher':
                if write_attns:
                    proba, attns = model.forward_teacher(batch, return_attns=write_attns)
                    attns_tot += attns
                else: 
                    proba = model.forward_teacher(batch, return_attns=write_attns)


            elif mode == 'student':
                output = model(batch)
                proba = F.softmax(output, dim=-1)
            else:
                raise NotImplementedError

            probas.append(proba.cpu().numpy())

        if write_attns:
            print('WRITING ATTENTIONS')
            out_fp = os.path.join(os.getenv('AMLT_OUTPUT_DIR', 'TEST_OUT'), 'attns.tsv')
            if os.path.exists(out_fp):
                os.remove(out_fp)

            print(attns_tot)
            with open(out_fp, 'w') as f:
                f.write(str(attns_tot))

        return np.vstack(probas)



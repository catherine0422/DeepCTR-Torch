# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,weichenswc@163.com

"""
from __future__ import print_function

import time
from collections import OrderedDict

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader
from sklearn.metrics import *
from torch.autograd import Variable
from tqdm import tqdm

try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList

from ..inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup, build_input_values_index, varlen_embedding_lookup_from_value_list
from ..layers import PredictionLayer
from ..layers.utils import slice_arrays
from ..callbacks import History,ModelCheckpoint
from ..attacks.utils import *
from ..attacks import FGSM
from ..datasets import NpyDataset


class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)

        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, 1, sparse=True) for feat in
        #              self.sparse_feature_columns}
        #         )
        # .to("cuda:1")
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1).to(
                device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def one_hot_value_from_feature_columns(self, X):
        sparse_value_list = [
            F.one_hot(X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].to(torch.int64),
                      num_classes=feat.vocabulary_size).squeeze(dim=1).float() for
            feat in self.sparse_feature_columns]
        var_len_sparse_value_list = [
            F.one_hot(X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].to(torch.int64),
                      num_classes=feat.vocabulary_size).squeeze(dim=1).float() for
            feat in self.varlen_sparse_feature_columns
        ]
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]
        return dense_value_list, sparse_value_list, var_len_sparse_value_list

    def input_from_value_lists(self, X, dense_value_list, sparse_value_list, var_len_sparse_value_list):

        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](sparse_value_list[i]) for
                                 i, feat in enumerate(self.sparse_feature_columns)]

        sequence_embed_dict = varlen_embedding_lookup_from_value_list(var_len_sparse_value_list, self.embedding_dict,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list

        return sparse_embedding_list, dense_value_list

    def input_from_feature_columns(self, X):
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list

        return sparse_embedding_list,dense_value_list


    def use_embeddings(self, sparse_embedding_cat, dense_value_cat, sparse_feat_refine_weight=None):
        device = sparse_embedding_cat.device if sparse_embedding_cat is not None else dense_value_cat.device
        linear_logit = torch.tensor([0]).to(device)
        if sparse_embedding_cat is not None:
            if sparse_feat_refine_weight is not None:
                # w_{x,i}=m_{x,i} * w_i (in IFM and DIFM)
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=True)
            linear_logit = sparse_feat_logit + linear_logit
        if dense_value_cat is not None:
            dense_value_logit = dense_value_cat.matmul(self.weight)
            linear_logit = dense_value_logit + linear_logit

        return linear_logit

    def forward(self, X, sparse_feat_refine_weight=None):

        # dense_value_list, sparse_value_list, var_len_sparse_value_list = self.one_hot_value_from_feature_columns(X)

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X)

        linear_logit = self.use_embeddings(sparse_embedding_list, dense_value_list, sparse_feat_refine_weight)

        return linear_logit


class BaseModel(nn.Module):
    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):

        super(BaseModel, self).__init__()
        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)
        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=True) for feat in
        #              self.dnn_feature_columns}
        #         )

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device)

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.out = PredictionLayer(task, )
        self.to(device)

        # parameters for callbacks
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14
        self.history = History()

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

        self.flatten_value_index = build_input_values_index(self.dense_feature_columns, self.sparse_feature_columns,
                                                            self.varlen_sparse_feature_columns)
        self.embedding_size = self.get_embedding_size()

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, adv_type=None, attacker=FGSM(), lam=1,
            eval_batch_size=256, count=False, p_data_sample=1):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, 2 or 3. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch,3 = one progress bar and one line per epoch
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`
        :param adv_type: `None` or str in [normal, free, free_lb, trades]. Whether to use adv training or not, and the type of adv train.
        :param attacker: Attack object. Perform adversarial attack in adv training.
        :param lam: Float. Proportion between adv and origin samples in adv training.
        :param normalized_attack: Int. 0, 1. Whether to normalize attack. 0 = False. 1 = Normalize according to the whole training data.

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """

        if count and (attacker.attack != 'ONE_CLASS' or adv_type != 'normal'):
            raise ValueError('Count is only implemented under one class attack, current attacker type is: ',
                             attacker.attack, ', adv type is: ', adv_type)

        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []

        if type(x) == str:
            # x is a file name, use npy dataset
            train_tensor_data = NpyDataset(x, y, p_sample=p_data_sample)
        else:
            for i in range(len(x)):
                if len(x[i].shape) == 1:
                    x[i] = np.expand_dims(x[i], axis=1)
            train_tensor_data = Data.TensorDataset(
                torch.from_numpy(
                    np.concatenate(x, axis=-1)),
                torch.from_numpy(y))

        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False
        save_freq = steps_per_epoch
        for callback in callbacks.callbacks:
            if type(callback) == ModelCheckpoint:
                if type(callback.save_freq) != str:
                    save_freq = callback.save_freq
        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        if adv_type is not None and adv_type in ['free', 'free_new']:
            global_noise_data = []
        total_max_idx_count = {}
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            batch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))
            try:
                with tqdm(enumerate(train_loader), disable=verbose not in [1, 3], leave=False) as t:
                    for steps, (x_train, y_train) in t:

                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        if adv_type is not None:
                            if adv_type == 'free':
                                # Adversarial training for free
                                if not attacker or attacker.attack != 'PGD':
                                    raise ValueError('A PGD attacker should be implemented when using Free method.')
                                eps = attacker.eps
                                free_steps = attacker.steps

                                for i in range(free_steps):
                                    # prediction on original sample
                                    optim.zero_grad()
                                    y_pred = model(x).squeeze()
                                    loss = loss_func(y_pred, y.squeeze(), reduction='sum')

                                    # Ascend on the global noise
                                    original_embeddings = model.get_embeddings(x,
                                                                               part_specified=attacker.part_specified)
                                    if len(global_noise_data) == 0:
                                        # initialize noise data
                                        global_noise_data = apply2nestLists(
                                            lambda x: torch.zeros_like(x).to(self.device).detach(), original_embeddings)
                                    noise_batch = apply2nestLists(
                                        lambda x: Variable(x[:y.size(0)], requires_grad=True).to(self.device),
                                        global_noise_data)
                                    if attacker.normalized:
                                        var_list = apply2nestLists(lambda x: torch.var(x.detach(), dim=0),
                                                                   original_embeddings)
                                        noise_batch = denormalize_data(noise_batch, var_list)
                                        noise_batch = apply2nestLists(
                                            lambda x: Variable(x, requires_grad=True).to(self.device), noise_batch)
                                    adv_embeddings = add_nestLists(original_embeddings, noise_batch)
                                    adv_pred = model.use_embeddings(adv_embeddings).squeeze()
                                    adv_loss = loss_func(adv_pred, y.squeeze(), reduction='sum')
                                    reg_loss = self.get_regularization_loss()
                                    total_loss = loss + reg_loss + self.aux_loss + lam * adv_loss

                                    loss_epoch += loss.item() / free_steps
                                    total_loss_epoch += total_loss.item() / free_steps

                                    # compute gradient
                                    total_loss.backward()

                                    deltas = delta_step(noise_batch, eps, need_get_grad=True)
                                    global_noise_data = add_nestLists(deltas, global_noise_data)
                                    global_noise_data = clamp_step(global_noise_data, eps)

                                    optim.step()

                            elif adv_type == 'free_lb':
                                # Adversarial training for FREELB
                                if not attacker or attacker.attack != 'PGD':
                                    raise ValueError('A PGD attacker should be implemented when using Free-LB method.')
                                eps = attacker.eps
                                alpha = attacker.alpha
                                free_lb_steps = attacker.steps
                                random_start = attacker.random_start
                                global_noise_data = []
                                optim.zero_grad()

                                # prediction on adv sample
                                for i in range(free_lb_steps):

                                    # prediction on original sample
                                    y_pred = model(x).squeeze()
                                    loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                                    loss_epoch += loss.item() / free_lb_steps

                                    # Ascend on the global noise
                                    original_embeddings = model.get_embeddings(x,
                                                                               part_specified=attacker.part_specified)
                                    if len(global_noise_data) == 0:
                                        # initialize noise data
                                        if random_start:
                                            global_noise_data = apply2nestLists(
                                                lambda x: torch.empty_like(x).uniform_(-eps, eps).detach(),
                                                original_embeddings)
                                        else:
                                            global_noise_data = apply2nestLists(
                                                lambda x: torch.zeros_like(x).to(self.device).detach(),
                                                original_embeddings)
                                    noise_batch = apply2nestLists(
                                        lambda x: Variable(x[:y.size(0)], requires_grad=True).to(self.device),
                                        global_noise_data)
                                    if attacker.normalized > 0:
                                        var_list = apply2nestLists(lambda x: torch.var(x.detach(), dim=0),
                                                                   original_embeddings)
                                        noise_batch = denormalize_data(noise_batch, var_list)
                                        noise_batch = apply2nestLists(
                                            lambda x: Variable(x, requires_grad=True).to(self.device), noise_batch)
                                    adv_embeddings = add_nestLists(original_embeddings, noise_batch)
                                    adv_pred = model.use_embeddings(adv_embeddings).squeeze()
                                    reg_loss = self.get_regularization_loss()
                                    adv_loss = loss_func(adv_pred, y.squeeze(), reduction='sum')
                                    total_loss = loss + reg_loss + self.aux_loss + lam * adv_loss

                                    total_loss_epoch += total_loss.item() / free_lb_steps

                                    # compute gradient
                                    total_loss.backward()

                                    deltas = delta_step(noise_batch, alpha, need_get_grad=True)
                                    global_noise_data = add_nestLists(deltas, global_noise_data)
                                    global_noise_data = clamp_step(global_noise_data, eps)

                                for param in model.parameters():
                                    param.grad /= free_lb_steps
                                optim.step()

                            elif adv_type in ['normal', 'trades']:
                                # prediction on original sample
                                optim.zero_grad()
                                y_pred = model(x)
                                loss = loss_func(y_pred.squeeze(), y.squeeze(), reduction='sum')

                                # prediction on adv sample
                                adv_loss = 0
                                if adv_type == 'trades':
                                    attacker.set_trades_mode(True)
                                if attacker.attack == 'ONE_CLASS':
                                    if count:
                                        adv_value_lists, max_idx_count = attacker(x, y, model, count=count)
                                        total_max_idx_count = append_counts(total_max_idx_count, max_idx_count)
                                    else:
                                        adv_value_lists = attacker(x, y, model)
                                    adv_pred = model.use_one_hot_values(x, adv_value_lists)
                                else:
                                    original_embeddings = model.get_embeddings(x, part_specified=attacker.part_specified)
                                    if attacker.normalized:
                                        var_list = apply2nestLists(lambda x: torch.var(x.detach(), dim=0),
                                                                   original_embeddings)
                                        attacker.set_normalize_params(var_list)
                                    deltas = attacker(x, y, model)
                                    adv_embeddings = add_nestLists(original_embeddings, deltas)
                                    adv_pred = model.use_embeddings(adv_embeddings)
                                    del adv_embeddings
                                if adv_type == 'normal':
                                    adv_loss = loss_func(adv_pred.squeeze(), y.squeeze(), reduction='sum')
                                elif adv_type == 'trades':
                                    adv_loss = trades_loss(y_pred, adv_pred)

                                y_pred = y_pred.squeeze()
                                adv_pred = adv_pred.squeeze()
                                reg_loss = self.get_regularization_loss()
                                total_loss = loss + reg_loss + self.aux_loss + lam * adv_loss

                                loss_epoch += loss.item()
                                total_loss_epoch += total_loss.item()
                                total_loss.backward()
                                optim.step()

                                del original_embeddings
                                del total_loss
                                torch.cuda.empty_cache()

                                attacker.set_trades_mode(False)

                            elif adv_type not in ['normal', 'free', 'free_lb', 'free_new', 'trades']:
                                raise NotImplementedError(
                                    f'adversarial training type not defined: {adv_type}, should be within' +
                                    f'["normal","free","free_lb","free_new", "trades"]')
                        else:
                            ## normal traning
                            optim.zero_grad()
                            y_pred = model(x).squeeze()
                            loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                            reg_loss = self.get_regularization_loss()
                            total_loss = loss + reg_loss + self.aux_loss

                            loss_epoch += loss.item()
                            total_loss_epoch += total_loss.item()
                            total_loss.backward()
                            optim.step()


                        if 0 in y and 1 in y:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))
                                if adv_type is not None:
                                    name_adv = "adv_" + name
                                    if name_adv not in train_result:
                                        train_result[name_adv] = []
                                    train_result[name_adv].append(metric_fun(
                                        y.cpu().data.numpy(), adv_pred.cpu().data.numpy().astype("float64")))
                        epoch_end = (steps+1)==steps_per_epoch
                        if (steps+1) % save_freq == 0 or epoch_end:
                            batch_logs["loss"] = total_loss_epoch / ((steps+1)*batch_size) if not epoch_end else total_loss_epoch / sample_num
                            for name, result in train_result.items():
                                batch_logs[name] = np.sum(result) / (steps+1)
                            if do_validation:
                                eval_result = self.evaluate(val_x, val_y, eval_batch_size)
                                for name, result in eval_result.items():
                                    batch_logs["val_" + name] = result
                                if adv_type is not None:
                                    adv_eval_result = self.adv_attack(val_x, val_y, attacker,
                                                                      batch_size=eval_batch_size)
                                    for name, result in adv_eval_result.items():
                                        batch_logs["val_adv_" + name] = result
                            for k, v in batch_logs.items():
                                epoch_logs.setdefault(k, []).append(v)

                            # verbose
                            if verbose > 1:
                                current_time = time.time()
                                batch_time = int(current_time - start_time)
                                eval_str = f"{batch_time}s [{steps+1:05d}/{steps_per_epoch:05d}] - loss: {batch_logs['loss']: .4f}"

                                for name in self.metrics:
                                    eval_str += " - " + name + \
                                                ": {0: .4f}".format(batch_logs[name])
                                    if adv_type is not None:
                                        name_adv = "adv_" + name
                                        eval_str += " - " + name_adv + \
                                                    ": {0: .4f}".format(batch_logs[name_adv])

                                if do_validation:
                                    eval_str += " \n "
                                    for name in self.metrics:
                                        eval_str += " - " + "val_" + name + \
                                                    ": {0: .4f}".format(batch_logs["val_" + name])
                                        if adv_type is not None:
                                            name_adv = "adv_" + name
                                            eval_str += " - " + "val_" + name_adv + \
                                                        ": {0: .4f}".format(batch_logs["val_" + name_adv])
                                print(eval_str)

                            callbacks.on_train_batch_end(steps+1, batch_logs)

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            epoch_time = int(time.time() - start_time)
            epoch_logs['epoch_time'] = epoch_time

            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()
        if count:
            return self.history, total_max_idx_count
        else:
            return self.history

    def evaluate(self, x, y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        if type(y) == str:
            y = np.load(y)
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        if type(x) == str:
            # x is a file name, use npy dataset
            tensor_data = NpyDataset(x)
        else:
            for i in range(len(x)):
                if len(x[i].shape) == 1:
                    x[i] = np.expand_dims(x[i], axis=1)
            tensor_data = Data.TensorDataset(
                torch.from_numpy(np.concatenate(x, axis=-1)))

        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

    def adv_attack(self, x, y, attacker, verbose=False, batch_size=256, count=False):
        r""" Apply adversarial attack on the data x, and return evaluations of the model under attack

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :param attacker: Attack object.
        :return: dict of evaluations
        """
        if count and attacker.attack != 'ONE_CLASS':
            raise ValueError('Count should by compared with an one class attak, current attacker type is:',
                             attacker.attack)
        pred_ans, distortion, max_idx_count = self.adv_pred(x, y, attacker, batch_size=batch_size, count=count,verbose=verbose)

        eval_result = {}
        if type(y) == str:
            y = np.load(y)
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)

        eval_result['distortion'] = distortion.item()
        if verbose:
            eval_str = ''
            for k, v in eval_result.items():
                eval_str += f'{k} = {v:.6f}, ' if type(v) != str else f'{k} = {v}, '
            eval_str += 'attack = ' + str(attacker)
            print(eval_str)
        if count:
            return eval_result, max_idx_count
        else:
            return eval_result

    def adv_pred(self, x, y, attacker, batch_size=256, count=False, verbose = False):
        model = self.eval()
        if type(x) == str:
            if type(y) != str:
                raise ValueError('argument x and y should all be file name to use NpyDataset')
            else:
                tensor_data = NpyDataset(x, y)
        else:
            if isinstance(x, dict):
                x = [x[feature] for feature in self.feature_index]
            for i in range(len(x)):
                if len(x[i].shape) == 1:
                    x[i] = np.expand_dims(x[i], axis=1)
            tensor_data = Data.TensorDataset(
                torch.from_numpy(np.concatenate(x, axis=-1)),
                torch.from_numpy(y))

        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        distortion_sum = torch.tensor([0]).to(self.device).float()
        pred_ans = []
        total_max_idx_count = {}
        with torch.no_grad():
            for x, label in tqdm(test_loader, disable=int(verbose)<2, leave=False):
                x, label = x.to(self.device).float(), label.to(self.device).float()
                if attacker.attack != 'ONE_CLASS':
                    original_embeddings = model.get_embeddings(x, part_specified=attacker.part_specified)
                    if attacker.normalized:
                        var_list = apply2nestLists(lambda x: torch.var(x.detach(), dim=0),
                                                   original_embeddings)
                        attacker.set_normalize_params(var_list)
                    with torch.enable_grad():
                        deltas = attacker(x, label, model)
                    adv_embeddings = add_nestLists(original_embeddings, deltas)
                    distortion_sum += get_rmse(deltas)
                    pred_an = model.use_embeddings(adv_embeddings)
                else:
                    value_lists = model.get_one_hot_values(x)
                    if count:
                        with torch.enable_grad():
                            adv_value_lists, max_idx_count = attacker(x, label, model, value_lists=value_lists,
                                                                      count=count)
                        total_max_idx_count = append_counts(total_max_idx_count, max_idx_count)
                    else:
                        with torch.enable_grad():
                            adv_value_lists = attacker(x, label, model, value_lists=value_lists)
                    pred_an = model.use_one_hot_values(x, adv_value_lists)
                pred_ans.append(pred_an.cpu().data.numpy())
        pred_ans = np.concatenate(pred_ans).astype("float64")
        distortion = distortion_sum / len(test_loader)
        return pred_ans, distortion, total_max_idx_count

    def get_full_emb_lists(self, x, batch_size=256, part_specified=False):
        r""" Calculate the scale of embeddings of x

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return:
            -not part_specified:
                [dense_value_list, linear_sparse_embedding_list, sparse_embedding_list]
            -part_specified:
                [LR_dense_value_tensor, LR_linear_sparse_embedding_tensor, FM_sparse_embedding_tensor,
                               DNN_dense_value_tensor, DNN_sparse_embedding_tensor]
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        all_emb_lists = []

        def append_in_nest(x, y):
            x.append(y)
            return x

        for x in test_loader:
            x = x[0].to(self.device).float()
            embeddings = model.get_embeddings(x, part_specified=part_specified)
            batch_embeddings_list = apply2nestLists(lambda x: x.cpu().detach().numpy(),
                                                    embeddings)  ## turn tensor into list
            if len(all_emb_lists) > 0:
                all_emb_lists = apply2nestLists(append_in_nest, (all_emb_lists, batch_embeddings_list))
            else:
                all_emb_lists = apply2nestLists(lambda x: [x], batch_embeddings_list)
        all_emb_lists = apply2nestLists(lambda x: np.concatenate(x, axis=0), all_emb_lists)
        return all_emb_lists

    def compute_full_emb_var(self, x, batch_size=256, part_specified=False):
        all_emb_lists = self.get_full_emb_lists(x, batch_size, part_specified)
        all_emb_var = apply2nestLists(lambda x: torch.tensor(np.var(x, axis=0)).to(self.device), all_emb_lists)
        return all_emb_var

    def emb_size_list_from_feature_columns(self):
        sparse_feature_columns = self.sparse_feature_columns + self.varlen_sparse_feature_columns
        dense_feature_columns = self.dense_feature_columns

        sparse_emb_size = sum(feat.embedding_dim for feat in sparse_feature_columns)
        linear_sparse_emb_size = len(sparse_feature_columns)
        dense_value_emb_size = len(dense_feature_columns)

        return sparse_emb_size, linear_sparse_emb_size, dense_value_emb_size

    def one_hot_value_from_feature_columns(self, X, support_dense=True):

        if not support_dense and len(self.dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_value_list = [
            F.one_hot(X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].to(torch.int64),
                      num_classes=feat.vocabulary_size).float() for feat in self.sparse_feature_columns]

        var_len_sparse_value_list = [
            F.one_hot(X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].to(torch.int64),
                      num_classes=feat.vocabulary_size).float() for feat in self.varlen_sparse_feature_columns
        ]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        return dense_value_list, sparse_value_list, var_len_sparse_value_list

    def input_from_value_lists(self, dense_value_list, sparse_value_list, var_len_sparse_value_list, X,
                                   embedding_dict):

        sparse_embedding_list = [embedding_dict[feat.embedding_name](sparse_value_list[i]) for
                                 i, feat in enumerate(self.sparse_feature_columns)]

        sequence_embed_dict = varlen_embedding_lookup_from_value_list(var_len_sparse_value_list, self.embedding_dict,
                                                      self.varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               self.varlen_sparse_feature_columns, self.device)

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list


    def input_from_feature_columns(self, X,  embedding_dict, support_dense=True):

        if not support_dense and len(self.dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")


        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.varlen_sparse_feature_columns)

        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               self.varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = self.sparse_feature_columns + self.varlen_sparse_feature_columns
        dense_feature_columns = self.dense_feature_columns

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    def compile(self, optimizer,
                loss=None,
                metrics=None,
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                self.metrics_names.append(metric)
        return metrics_

    def _in_multi_worker_mode(self):
        # used for EarlyStopping in tf1.15
        return None

    def get_embedding_size(self, ):
        sparse_feature_columns = self.sparse_feature_columns + self.varlen_sparse_feature_columns
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]

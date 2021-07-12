# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""
import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input, combined_dnn_input_tensor
from ..layers import FM, DNN, concat_fun


class DeepFM(BaseModel):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0, emb_use_bn=False, emb_use_bn_simple=False,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)

        self.emb_use_bn = emb_use_bn
        self.emb_use_bn_simple = emb_use_bn_simple
        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        if self.emb_use_bn or self.emb_use_bn_simple:
            affine = not self.emb_use_bn_simple
            sparse_emb_size, linear_sparse_emb_size, dense_value_emb_size = self.emb_size_list_from_feature_columns(
                dnn_feature_columns)
            self.emb_bn_sparse = nn.BatchNorm1d(sparse_emb_size, affine = affine) if sparse_emb_size> 0 else None
            self.emb_bn_linear_sparse = nn.BatchNorm1d(linear_sparse_emb_size, affine = affine) if linear_sparse_emb_size> 0 else None
            self.emb_bn_dense = nn.BatchNorm1d(dense_value_emb_size, affine = affine) if dense_value_emb_size > 0 else None

        self.to(device)

    def get_embeddings(self, X, part_specified=False):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        linear_sparse_embedding_list, linear_dense_value_list = self.linear_model.input_from_feature_columns(X)

        sparse_embedding_tensor = concat_fun(sparse_embedding_list).squeeze()
        linear_sparse_embedding_tensor = concat_fun(linear_sparse_embedding_list).squeeze()
        dense_value_tensor = concat_fun(dense_value_list)

        if self.emb_use_bn or self.emb_use_bn_simple:
            if sparse_embedding_tensor is not None:
                sparse_embedding_tensor = self.emb_bn_sparse(sparse_embedding_tensor)
            if linear_sparse_embedding_tensor is not None:
                linear_sparse_embedding_tensor = self.emb_bn_linear_sparse(linear_sparse_embedding_tensor)
            if dense_value_tensor is not None:
                dense_value_tensor = self.emb_bn_dense(dense_value_tensor)

        if part_specified:
            DNN_sparse_embedding_tensor = FM_sparse_embedding_tensor = sparse_embedding_tensor
            LR_linear_sparse_embedding_tensor = linear_sparse_embedding_tensor
            DNN_dense_value_tensor = LR_dense_value_tensor = dense_value_tensor
            embedding_lists = [LR_dense_value_tensor, LR_linear_sparse_embedding_tensor, FM_sparse_embedding_tensor,
                               DNN_dense_value_tensor, DNN_sparse_embedding_tensor]
        else:
            embedding_lists = [dense_value_tensor, linear_sparse_embedding_tensor, sparse_embedding_tensor]

        return embedding_lists

    def use_embeddings(self, embedding_lists):
        part_specified = True if len(embedding_lists) > 3 else False
        if part_specified:
            LR_dense_value_tensor, LR_linear_sparse_embedding_tensor, FM_sparse_embedding_tensor,\
            DNN_dense_value_tensor, DNN_sparse_embedding_tensor = embedding_lists
        else:
            LR_dense_value_tensor, LR_linear_sparse_embedding_tensor, FM_sparse_embedding_tensor = embedding_lists
            DNN_dense_value_tensor, DNN_sparse_embedding_tensor = LR_dense_value_tensor, FM_sparse_embedding_tensor

        logit = self.linear_model.use_embeddings(LR_linear_sparse_embedding_tensor, LR_dense_value_tensor)

        if self.use_fm and FM_sparse_embedding_tensor is not None:
            # fm input size: [batch_size, sparse_feat_size, emb_size]
            fm_input = FM_sparse_embedding_tensor.view(FM_sparse_embedding_tensor.size(0), -1, self.embedding_size)
            logit += self.fm(fm_input)

        if self.use_dnn:
            dnn_input = combined_dnn_input_tensor(DNN_sparse_embedding_tensor, DNN_dense_value_tensor)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred

    def forward(self, X):

        embedding_lists = self.get_embeddings(X)

        y_pred = self.use_embeddings(embedding_lists)

        return y_pred

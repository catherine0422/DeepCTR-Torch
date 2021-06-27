# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
from deepctr_torch.attacks import FGSM

if __name__ == "__main__":
    data = pd.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = False
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    attacker = FGSM()
    print('Normal training')

    model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )


    history = model.fit(train_model_input, train[target].values, batch_size=32, epochs=10, verbose=1,
                        adv=False, validation_data=(test_model_input, test[target].values))
    eval = model.evaluate(test_model_input, test[target].values)
    print(eval)
    adv_eval = model.adv_attack(test_model_input, test[target].values, attacker)
    print(adv_eval)
    print()

    print('Adver training')
    model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )

    history = model.fit(train_model_input, train[target].values, batch_size=32, epochs=10, verbose=1,
                        adv=True, validation_data=(test_model_input, test[target].values))
    eval = model.evaluate(test_model_input, test[target].values)
    print(eval)
    adv_eval = model.adv_attack(test_model_input, test[target].values, attacker)
    print(adv_eval)

    # y = train[target].values
    # x = train_model_input
    # if isinstance(x, dict):
    #     x = [x[feature] for feature in x.keys()]
    # for i in range(len(x)):
    #     if len(x[i].shape) == 1:
    #         x[i] = np.expand_dims(x[i], axis=1)
    # x = np.concatenate(x, axis=-1)
    # train_tensor_data = Data.TensorDataset(
    #     torch.from_numpy(x),
    #     torch.from_numpy(y))
    # train_loader = DataLoader(dataset=train_tensor_data, shuffle=True, batch_size=256)
    #
    # it = iter(train_loader)
    # x_train, y_train = next(it)
    # x_train, y_train = x_train.to(device).float(), y_train.to(device).float()
    #
    # res = model.calculate_emb_scales(test_model_input, batch_size=64)
    # print(res)
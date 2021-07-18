import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr_torch.attacks.attacks.fgsm import FGSM
from deepctr_torch.attacks.attacks.gaussian import GAUSSIAN
from deepctr_torch.attacks.attacks.one_class import ONE_CLASS
from deepctr_torch.attacks.attacks.pgd import PGD
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import *


def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


def binarize(x):
        if x > 3:
            return 1
        elif x < 3:
            return 0
        else:
            return np.nan


if __name__ == "__main__":
    data = pd.read_csv("./movielens_sample.txt")
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip", ]
    target = ['rating']


    data['rating'] = data['rating'].apply(lambda x: binarize(x))
    data = data.dropna()

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # preprocess the sequence feature

    key2index = {}
    genres_list = list(map(split, data['genres'].values))
    genres_length = np.array(list(map(len, genres_list)))
    max_len = max(genres_length)
    # Notice : padding=`post`
    genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )

    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                              for feat in sparse_features]

    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
        key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean')]  # Notice : value 0 is for padding for sequence input feature

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    data['genres'] = genres_list.tolist()
    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    train_model_input['genres'] = np.stack(train_model_input['genres'].values)
    test_model_input['genres'] = np.stack(test_model_input['genres'].values)
    validate_data = (test_model_input,test[target].values)

    # 4.Define Model,compile and train

    device = 'cpu'
    use_cuda = False
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device)

    model.compile("adam", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"],)

    attacker = ONE_CLASS(mask=[(1,0),(1,1)])
    # attacker = PGD(eps=0.1)
    history,idx_count = model.fit(train_model_input, train[target].values, batch_size=32, epochs=10, verbose=2,
                        validation_data=validate_data, adv_type='normal',attacker=attacker,count=True)
    torch.save(model,'movie_model.pth')
    # model = torch.load('movie_model.pth')
    # eps = 40
    # var_list = model.compute_full_emb_var(test_model_input)
    # attacker = ONE_CLASS()
    # attacker.set_bias(var_list)

    res,idx_count = model.adv_attack(train_model_input, train[target].values, attacker,verbose=True,count=True)
    print(idx_count)
    #
    # attacker = FGSM(eps=eps)
    # attacker.set_bias(var_list)
    # res = model.adv_attack(test_model_input,test[target].values,attacker,verbose=True)
    #
    # attacker = GAUSSIAN(eps=eps)
    # attacker.set_bias(var_list)
    # res = model.adv_attack(test_model_input,test[target].values,attacker,verbose=True)
    #
    # attacker = PGD(eps=eps,steps=10)
    # res = model.adv_attack(test_model_input,test[target].values,attacker,verbose=True)
    #
    # attacker = FGSM(eps=eps)
    # res = model.adv_attack(test_model_input,test[target].values,attacker,verbose=True)
    #
    # attacker = GAUSSIAN(eps=eps)
    # res = model.adv_attack(test_model_input,test[target].values,attacker,verbose=True)



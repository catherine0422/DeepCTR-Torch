import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from ..attack import Attack
from ..utils import *


class ONE_CLASS(Attack):
    """
    Change one class that has the biggest gradient.

    Examples::
    --------
        >>> attack = deepctr_torch.attacks.ONE_CLASS()
        >>> adv_samples = attack(samples, labels, model)

    """

    def __init__(self, mask =[], random=False, trades=False):
        """
        Argument:
        --------
            - mask: list of (type index, tensor index), to indicate the feature to be masked in attacking
        Shape:
        -------
            - samples: :math:`(N, L)` where `N = number of batches`, `L = lenth of data`.
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`[[(N, 1, E)],[(N, 1, 1)],[(N, 1)]]`. where `E = embedding size`
                    The first element is the delta of sparse embeddings list, the second is the delta
                    of linear sparse embedding list, the last is the delta of dense value.

        """
        super(ONE_CLASS, self).__init__("ONE_CLASS")
        self.random=random
        self.mask=mask

    def attack_value_lists(self, adv_value_lists, i, type_idx, tensor_idx, ele_idx):

        if type_idx == 2:
            length_row = adv_value_lists[type_idx][tensor_idx][i].size(-1)
            row_idx = ele_idx // length_row - 1
            col_idx = ele_idx % length_row - 1
            delidx = adv_value_lists[type_idx][tensor_idx][i][row_idx] != 0
            adv_value_lists[type_idx][tensor_idx][i][row_idx][delidx] = 0
            adv_value_lists[type_idx][tensor_idx][i][row_idx][col_idx] = 1
        elif type_idx == 1:
            delidx = adv_value_lists[type_idx][tensor_idx][i][0] != 0
            adv_value_lists[type_idx][tensor_idx][i][0][delidx] = 0
            adv_value_lists[type_idx][tensor_idx][i][0][ele_idx] = 1
        else:
            adv_value_lists[type_idx][tensor_idx][i][0] = adv_value_lists[type_idx][tensor_idx][i][0] + 1

        return adv_value_lists

    def forward(self, x, labels, model, value_lists=None, count=False):
        """
        Overridden.
        """
        max_idx_count = {}
        if value_lists is None:
            value_lists = model.get_one_hot_values(x)

        if self.random:
            adv_value_lists = apply2nestLists_2layer(lambda x: x.detach().clone().requires_grad_(False),
                                                 value_lists)
            value_dict = model.flatten_value_index.copy()
            for (type_idx, tensor_idx) in self.mask:
                for k,v in value_dict.items():
                    if v[2] == type_idx and v[3] == tensor_idx:
                        del value_dict[k]
                        break
            for i in range(x.size(0)):
                feature_names=list(value_dict.keys())
                attack_feature_idx = np.random.randint(len(feature_names))
                feature_name = feature_names[attack_feature_idx]
                start_pos, end_pos, type_idx, tensor_idx = value_dict[feature_name]
                ele_idx = np.random.randint(end_pos-start_pos)
                adv_value_lists = self.attack_value_lists(adv_value_lists, i, type_idx, tensor_idx, ele_idx)
                if count:
                    max_idx_count = add_count(max_idx_count, (type_idx, tensor_idx, ele_idx))

        else:
            training_mode = model.training

            if self._training_mode:
                model.train()
            else:
                model.eval()

            adv_value_lists = apply2nestLists_2layer(lambda x: Variable(x, requires_grad=True), value_lists)
            pred = model.use_one_hot_values(x, adv_value_lists)
            loss_fct = nn.BCELoss()
            loss = loss_fct(pred, labels)
            loss.backward()
            grad_list = apply2nestLists_2layer(lambda x: x.grad, adv_value_lists)
            for (type_idx, tensor_idx) in self.mask:
                grad_list[type_idx][tensor_idx] -= float('inf')

            flatten_grad_tensor = flatten_grad_list(grad_list)
            max_index_list = torch.argmax(flatten_grad_tensor,dim=-1)

            value_dict = model.flatten_value_index # feat name: (start pos, end pos, type index, tensor index)
            adv_value_lists = apply2nestLists_2layer(lambda x: x.detach().clone().requires_grad_(False), adv_value_lists)
            for i in range(max_index_list.size(0)):
                max_index = max_index_list[i].item()
                for (start_pos,end_pos,type_idx,tensor_idx) in value_dict.values():
                    if max_index < end_pos:
                        ele_idx = max_index-start_pos
                        adv_value_lists = self.attack_value_lists(adv_value_lists, i, type_idx, tensor_idx, ele_idx)
                        if count:
                            max_idx_count = add_count(max_idx_count, (type_idx, tensor_idx, ele_idx))
                        break

            if training_mode:
                model.train()
        if count:
            return adv_value_lists, max_idx_count
        else:
            return adv_value_lists


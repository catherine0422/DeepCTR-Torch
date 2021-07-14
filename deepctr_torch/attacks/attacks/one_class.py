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

    def __init__(self, trades=False):
        """
        Shape:
        -------
            - samples: :math:`(N, L)` where `N = number of batches`, `L = lenth of data`.
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`[[(N, 1, E)],[(N, 1, 1)],[(N, 1)]]`. where `E = embedding size`
                    The first element is the delta of sparse embeddings list, the second is the delta
                    of linear sparse embedding list, the last is the delta of dense value.

        """
        super(ONE_CLASS, self).__init__("ONE_CLASS")


    def forward(self, x, labels, model, value_lists=None):
        """
        Overridden.
        """
        training_mode = model.training

        if self._training_mode:
            model.train()
        else:
            model.eval()
        if value_lists is None:
            value_lists = model.get_one_hot_values(x)
        adv_value_lists = apply2nestLists_2layer(lambda x: Variable(x, requires_grad=True), value_lists)
        pred = model.use_one_hot_values(x, adv_value_lists)
        loss_fct = nn.BCELoss()
        loss = loss_fct(pred, labels)
        loss.backward()
        grad_list = apply2nestLists_2layer(lambda x: x.grad, adv_value_lists)
        tensor_max_value_list = []
        tensor_max_idx_list = []
        ele_max_idx_lists = []
        for type_grad_list in grad_list:
          if len(type_grad_list) > 0:
            batch_size = type_grad_list[0].shape[0]
            break
        for type_grad_list in grad_list:
          if len(type_grad_list) > 0:
            ele_max_value_list = []
            ele_max_idx_list = []
            for grad_tensor in type_grad_list:
              grad_tensor = grad_tensor.detach().cpu().numpy()
              if grad_tensor.ndim > 2:
                grad_tensor = grad_tensor.reshape(batch_size, -1)
              ## 一个tensor中最大值 [one-hot size]选一
              ele_max_value= np.expand_dims(np.max(grad_tensor, axis=1),axis=1)
              ele_max_idx = np.argmax(grad_tensor, axis=1)
              ele_max_value_list.append(ele_max_value)
              ele_max_idx_list.append(ele_max_idx)
            ele_max_idx_lists.append(ele_max_idx_list)
            tensor_max_value_tensor = np.concatenate(ele_max_value_list, axis=1)
            ## 一个type中最大值 [number of tensors]选一
            tensor_max_value= np.expand_dims(np.max(tensor_max_value_tensor, axis=1),axis=1)
            tensor_max_idx = np.argmax(tensor_max_value_tensor, axis=1)
            tensor_max_value_list.append(tensor_max_value)
            tensor_max_idx_list.append(tensor_max_idx)
          else:
            ele_max_idx_lists.append(np.ones(batch_size)*(-1))
            tensor_max_value_list.append(np.ones((batch_size,1))*-float('inf'))
            tensor_max_idx_list.append(np.ones(batch_size)*(-1))
        ## 所有type中最大值 [number of types]选一
        type_max_value_tensor = np.concatenate(tensor_max_value_list, axis=-1)
        type_max_idx = np.argmax(type_max_value_tensor, axis=1)

        adv_value_lists = apply2nestLists_2layer(lambda x:x.detach().clone().requires_grad_(False), adv_value_lists)
        for i in range(batch_size):
          type_idx = type_max_idx[i]
          tensor_idx = tensor_max_idx_list[type_idx][i]
          ele_idx = ele_max_idx_lists[type_idx][tensor_idx][i]
          if type_idx == 2:
            length_row = adv_value_lists[type_idx][tensor_idx][i].size(-1)
            row_idx = ele_idx // length_row - 1
            col_idx = ele_idx % length_row - 1
            delidx = adv_value_lists[type_idx][tensor_idx][i][row_idx] != 0
            adv_value_lists[type_idx][tensor_idx][i][row_idx][delidx] = 0
            adv_value_lists[type_idx][tensor_idx][i][row_idx][col_idx] = 1
          elif type_idx == 1:
            delidx = adv_value_lists[type_idx][tensor_idx][i] != 0
            adv_value_lists[type_idx][tensor_idx][i][delidx] = 0
            adv_value_lists[type_idx][tensor_idx][i][ele_idx] = 1
          else:
            raise NotImplementedError('dense error not implemented')

        if training_mode:
            model.train()

        return adv_value_lists

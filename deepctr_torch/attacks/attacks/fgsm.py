import torch
import torch.nn as nn

from ..attack import Attack
from ..utils import *


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFAULT: 0.001)

    Shape:
        - samples: :math:`(N, L)` where `N = number of batches`, `L = lenth of data`.
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`[[(N, 1, E)],[(N, 1, 1)],[(N, 1)]]`. where `E = embedding size`
                The first element is the delta of sparse embeddings list, the second is the delta
                of linear sparse embedding list, the last is the delta of dense value.

    Examples::
        >>> attack = deepctr_torch.attacks.FGSM(eps=0.001)
        >>> adv_samples = attack(samples, labels)

    """

    def __init__(self, eps=0.001):
        super(FGSM, self).__init__("FGSM")
        self.eps = eps

    def forward(self, samples, labels, model):
        r"""
        Overridden.
        """
        training_mode = model.training

        if self._training_mode:
            model.train()
        else:
            model.eval()

        original_embeddings = model.get_embeddings(samples)
        original_embeddings = apply2nestLists(lambda x: clone_embs(x, model.device), original_embeddings)
        loss_fct = nn.BCELoss()
        pred = model.use_embeddings(*original_embeddings)
        cost = loss_fct(pred, labels)

        grads = apply2nestLists(lambda x: get_grad(x, cost), original_embeddings)
        deltas = apply2nestLists(lambda x: self.eps * x.sign(), grads)

        if training_mode:
            model.train()

        return deltas
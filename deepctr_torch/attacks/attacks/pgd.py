import numpy as np
import torch.nn as nn
from ..attack import Attack
from ..utils import *


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Examples::
    --------
        >>> attack = deepctr_torch.attacks.PGD(eps=0.001)
        >>> adv_samples = attack(samples, labels)

    """

    def __init__(self, eps=0.001, alpha=0.002 / 7, steps=7, random_start=False, trades=False, part_specified=False):
        """
        Arguments:
        --------
            :param model:  (nn.Module)
                model to attack.

            :param eps:  (list of float or float)
                maximum perturbation. (DEFAULT: 0.3) If list, specified for each embedding.

            :param alpha: (float or list of float)
                step size. (DEFAULT: 2/255) If list, specified for each embedding.

            :param steps: (int)
                number of steps. (DEFAULT: 40)

            :param random_start: (bool)
                using random initialization of delta. (DEFAULT: False)

            :param part_specified: (Boolean)
                whether to create specified delta for different par of the model
                    - if True: LR_dense, LR_linear_sparse, FM_sparse, DNN_dense, DNN_sparse
                    - if False: dense, linear_sparse, sparse

        Shape:
        -------
            - samples: :math:`(N, L)` where `N = number of batches`, `L = lenth of data`.
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`[[(N, 1, E)],[(N, 1, 1)],[(N, 1)]]`. where `E = embedding size`
                    The first element is the delta of sparse embeddings list, the second is the delta
                    of linear sparse embedding list, the last is the delta of dense value.
        """
        super(PGD, self).__init__("PGD", part_specified)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.trades = trades

    def forward(self, samples, labels, model):
        r"""
        Overridden.
        """
        training_mode = model.training

        if self._training_mode:
            model.train()
        else:
            model.eval()

        original_embeddings = model.get_embeddings(samples, self.part_specified)
        if self.trades:
            pred = model.use_embeddings(original_embeddings)
            loss_fct = nn.KLDivLoss(reduction='batchmean')
        else:
            loss_fct = nn.BCELoss()
        adv_embeddings = apply2nestLists(lambda x: x.clone().detach().to(model.device), original_embeddings)

        if self.random_start or self.trades:
            # Starting at a uniformly random point
            adv_embeddings = apply2nestLists(lambda x: x + torch.empty_like(x).uniform_(-self.eps, self.eps).detach(),
                                             adv_embeddings)

        for i in range(self.steps):

            adv_embeddings = apply2nestLists(lambda x: x.requires_grad_(True), adv_embeddings)
            adv_pred = model.use_embeddings(adv_embeddings)

            if not self.trades:
                cost = loss_fct(adv_pred, labels)
            else:
                cost = trades_loss(pred, adv_pred)

            grads = apply2nestLists(lambda x: get_grad(x, cost), adv_embeddings)
            deltas = delta_step(grads, self.alpha)
            global_distortion = apply2nestLists(lambda x, y, z: x + y - z,
                                                (adv_embeddings, deltas, original_embeddings))
            deltas = clamp_step(global_distortion, self.eps)
            adv_embeddings = apply2nestLists(lambda x, y: (x + y).detach(), (original_embeddings, deltas))

        if training_mode:
            model.train()

        return deltas

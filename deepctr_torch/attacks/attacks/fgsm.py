import torch.nn as nn
from ..attack import Attack
from ..utils import *


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Examples::
    --------
        >>> attack = deepctr_torch.attacks.FGSM(eps=0.001)
        >>> adv_samples = attack(samples, labels)

    """

    def __init__(self, eps=0.001, random_start=False, trades=False, part_specified=False, var_list = None, normalized = False):
        """
        Arguments:
        --------
            :param model: (nn.Module)
                model to attack.

            :param eps: (float or list of float)
                maximum perturbation. (DEFAULT: 0.001) If list, specified for each embedding.

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
        super(FGSM, self).__init__("FGSM", part_specified=part_specified, var_list = var_list, normalized = normalized)
        self.eps = eps
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

        original_embeddings = model.get_embeddings(samples, part_specified=self.part_specified)

        adv_embeddings = apply2nestLists(lambda x: x.clone().detach().to(model.device), original_embeddings)

        if self.random_start or self.trades:
            # Starting at a uniformly random point
            adv_embeddings = apply2nestLists(lambda x: x + torch.empty_like(x).uniform_(-self.eps, self.eps).detach(),
                                             adv_embeddings)

        adv_embeddings = apply2nestLists(lambda x: x.requires_grad_(True), adv_embeddings)
        adv_pred = model.use_embeddings(adv_embeddings)
        if not self.trades:
            loss_fct = nn.BCELoss()
            cost = loss_fct(adv_pred, labels)
        else:
            pred = model.use_embeddings(original_embeddings)
            cost = trades_loss(pred, adv_pred)

        grads = apply2nestLists(lambda x: get_grad(x, cost), adv_embeddings)
        deltas = delta_step(grads, self.eps)
        if self.normalized:
            deltas = denormalize_data(deltas, self.var_list, self.bias_eps)

        if training_mode:
            model.train()

        return deltas

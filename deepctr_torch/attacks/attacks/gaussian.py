import torch.nn as nn

from ..attack import Attack
from ..utils import *


class GAUSSIAN(Attack):
    r"""
    Gaussian distortions

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
        >>> attack = deepctr_torch.attacks.GAUSSIAN(eps=0.001)
        >>> adv_samples = attack(samples, labels)

    """

    def __init__(self, eps=0.001, part_specified=False,var_list = None, normalized = False):
        super(GAUSSIAN, self).__init__("GAUSSIAN", part_specified=part_specified, var_list = var_list, normalized = normalized)
        self.eps = eps

    def forward(self, samples, labels, model, value_lists=None):
        r"""
        Overridden.
        """
        training_mode = model.training

        if self._training_mode:
            model.train()
        else:
            model.eval()

        original_embeddings = model.get_embeddings(samples, part_specified=self.part_specified, value_lists=value_lists)
        f = lambda x,y: torch.normal(mean=0, std=x, size=y.size()).to(model.device) if x>0 else torch.zeros_like(y).to(model.device)
        deltas = func_detect_arg_type(f,self.eps,original_embeddings)
        if self.normalized:
            deltas = denormalize_data(deltas,self.var_list, self.bias_eps)

        if training_mode:
            model.train()

        return deltas
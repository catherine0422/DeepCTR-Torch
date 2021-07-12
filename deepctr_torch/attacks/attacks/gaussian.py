import numpy as np
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

    def __init__(self, eps=0.001, part_specified=False):
        super(GAUSSIAN, self).__init__("GAUSSIAN", part_specified)
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

        original_embeddings = model.get_embeddings(samples, self.part_specified)
        if type(self.eps) not in [list, np.ndarray, tuple]:
            if len(self.eps) != len(original_embeddings):
                raise ValueError(
                    f'number of clamp values dosen\'t fit the number of embeddings: {len(self.eps)} != {len(original_embeddings)}')
            deltas = apply2nestLists(lambda x: torch.normal(mean=0, std=self.eps, size=x.size()).to(model.device),
                                 original_embeddings)
        else:
            deltas = apply2nestLists(lambda x,y: torch.normal(mean=0, std=y, size=x.size()).to(model.device),
                                     (original_embeddings, self.eps))

        if training_mode:
            model.train()

        return deltas
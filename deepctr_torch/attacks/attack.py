import torch


class Attack(object):
    r"""
    Base class for all attacks.
    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
    """

    def __init__(self, name, part_specified = False):
        r"""
        Initializes internal attack state.
        Arguments:
            name (str): name of attack.
        """

        self.attack = name
        self._training_mode = False
        self.part_specified = part_specified

    def set_trades_mode(self, trades):
        self.trades = trades

    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def set_training_mode(self, flag):
        r"""
        Set training mode during attack process.
        Arguments:
            flag (bool): True for using training mode during attack process.
        """
        self._training_mode = flag

    def __str__(self):
        info = self.__dict__.copy()

        del_keys = ['attack']

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        return self.attack + "(" + ', '.join(f'{k}={v}' for k, v in info.items()) + ")"

    def __call__(self, *input, **kwargs):

        deltas = self.forward(*input, **kwargs)

        return deltas
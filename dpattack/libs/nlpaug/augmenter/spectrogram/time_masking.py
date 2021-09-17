"""
    Augmenter that apply time based masking to spectrogram input.
"""

from dpattack.libs.nlpaug import SpectrogramAugmenter
from dpattack.libs.nlpaug import Action
from dpattack.libs import nlpaug as nms


class TimeMaskingAug(SpectrogramAugmenter):
    """
    Augmenter that mask spectrogram based on time by random values.

    :param int mask_factor: Value between 0 and mask_factor will be picked randomly.
        Mask range will be between [0, tau - master_factor) while tau is time range of input.
    :param str name: Name of this augmenter

    >>> from dpattack.libs import nlpaug as nas
    >>> aug = nas.TimeMaskingAug(mask_factor=80)
    """

    def __init__(self, mask_factor, name='TimeMasking_Aug', verbose=0):
        super(TimeMaskingAug, self).__init__(
            action=Action.SUBSTITUTE, name=name, aug_p=1, aug_min=0.3, verbose=verbose)

        self.model = self.get_model(mask_factor)

    def substitute(self, data):
        return self.model.mask(data)

    @classmethod
    def get_model(cls, mask_factor):
        return nms.TimeMasking(mask_factor)

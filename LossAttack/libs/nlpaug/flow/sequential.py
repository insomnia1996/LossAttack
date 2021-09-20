"""
    Flow that apply augmentation sequentially.
"""

from LossAttack.libs.nlpaug import Action
from LossAttack.libs.nlpaug import Pipeline


class Sequential(Pipeline):
    """
    Flow that apply augmenters sequentially.

    :param list flow: list of flow or augmenter
    :param str name: Name of this augmenter

    >>> from LossAttack.libs import nlpaug as naf, nlpaug as nac, nlpaug as naw
    >>> flow = naf.Sequential([nac.RandomCharAug(), naw.RandomWordAug()])
    """

    def __init__(self, flow=None, name='Sequential_Pipeline', verbose=0):
        Pipeline.__init__(self, name=name, action=Action.SEQUENTIAL,
                          flow=flow, aug_min=-1, aug_p=1, verbose=verbose)

    def draw(self):
        return True

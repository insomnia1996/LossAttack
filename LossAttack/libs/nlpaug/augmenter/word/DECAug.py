"""
    Augmenter that apply operation (word level) to textual input based on contextual word embeddings.
"""

import string
from LossAttack.libs.nlpaug.augmenter.word import WordAugmenter
from LossAttack.cmds.DEC.model import DECParser
from LossAttack.libs.nlpaug.model import lang_models as nml
from LossAttack.libs.nlpaug.util import Action
from transformers import BartTokenizer, BartModel, BartConfig, BartForConditionalGeneration
BERT_MODEL = {}
XLNET_MODEL = {}


def init_bart_model(config):
    # Load models once at runtime

    bart_model = DECParser(config)
    bart_model.model.eval()


    return bart_model


def init_xlnet_model(model_path, device, force_reload=False, temperature=1.0, top_k=None, top_p=None):
    # Load models once at runtime

    global XLNET_MODEL
    if XLNET_MODEL and not force_reload:
        XLNET_MODEL.temperature = temperature
        XLNET_MODEL.top_k = top_k
        XLNET_MODEL.top_p = top_p
        return XLNET_MODEL

    xlnet_model = nml.XlNet(model_path, device=device, temperature=temperature, top_k=top_k, top_p=top_p)
    xlnet_model.model.eval()
    XLNET_MODEL = xlnet_model

    return xlnet_model


class DECAug(WordAugmenter):
    """
    Augmenter that leverage contextual word embeddings to find top n similar word for augmentation.

    :param str model_path: Model name or models path. It used transformers to load the models. Tested
        'bert-base-cased', 'xlnet-base-cased'.
    :param str action: Either 'insert or 'substitute'. If value is 'insert', a new word will be injected to random
        position according to contextual word embeddings calculation. If value is 'substitute', word will be replaced
        according to contextual embeddings calculation
    :param float temperature: Controlling randomness. Default value is 1 and lower temperature results in less random
        behavior
    :param int top_k: Controlling lucky draw pool. Top k score token will be used for augmentation. Larger k, more
        token can be used. Default value is 100. If value is None which means using all possible tokens.
    :param float top_p: Controlling lucky draw pool. Top p of cumulative probability will be removed. Larger p, more
        token can be used. Default value is None which means using all possible tokens.
    :param int aug_min: Minimum number of word will be augmented.
    :param float aug_p: Percentage of word will be augmented.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param bool skip_unknown_word: Do not substitute unknown word (e.g. AAAAAAAAAAA)
    :param str device: Use either cpu or gpu. Default value is None, it uses GPU if having. While possible values are
        'cuda' and 'cpu'.
    :param bool force_reload: Force reload the contextual word embeddings models to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param str name: Name of this augmenter

    >>> from LossAttack.libs import nlpaug as naw
    >>> aug = naw.ContextualWordEmbsAug()
    """

    def __init__(self, config, model_path='bert-base-cased', action="substitute", temperature=1.0, top_k=100, top_p=None,
                 name='ContextualWordEmbs_Aug', aug_min=1, aug_p=0.3, stopwords=None, skip_unknown_word=False,
                 device=None, force_reload=False, verbose=0):
        super().__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, tokenizer=None, stopwords=stopwords,
            verbose=verbose)
        self.model_path = model_path
        self.skip_unknown_word = skip_unknown_word
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self._init()
        self.model = self.get_model(config)
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir = "./data/pretrained/bart-base")

    def _init(self):
        if 'xlnet' in self.model_path:
            self.model_type = 'xlnet'
        elif 'bart' in self.model_path:
            self.model_type = 'bart'
        else:
            self.model_type = ''

    def skip_aug(self, token_idxes, tokens):
        if not self.skip_unknown_word:
            super().skip_aug(token_idxes, tokens)

        # text is split by " ". Need to join it and tokenizer it by models's tokenizer
        subwords = self.model.tokenizer.tokenize(' '.join(tokens))

        token2subword = {}
        subword_pos = 0

        for i, token in enumerate(tokens):
            token2subword[i] = []

            token2subword[i].append(subword_pos)
            subword_pos += 1

            for subword in subwords[subword_pos:]:
                if self.model_type in ['bert'] and self.model.SUBWORD_PREFIX in subword:
                    token2subword[i].append(subword_pos)
                    subword_pos += 1
                elif self.model_type in ['xlnet'] and self.model.SUBWORD_PREFIX not in subword and \
                        subword not in string.punctuation:
                    token2subword[i].append(subword_pos)
                    subword_pos += 1
                else:
                    break

        results = []
        for token_idx in token_idxes:
            # Skip if includes more than 1 subword. e.g. ESPP --> es ##pp (BERT), ESP --> ESP P (XLNet).
            # Avoid to substitute ESPP token
            if self.action == Action.SUBSTITUTE:
                if len(token2subword[token_idx]) == 1:
                    results.append(token_idx)
            else:
                results.append(token_idx)

        return results


    def substitute(self, data, aug_idxes = None, n=30):#aug_idxes为要攻击的位置列表
        # Pick target word for augmentation
        tokens = data.split(' ')
        if aug_idxes is None:
            aug_idxes = self._get_aug_idxes(tokens)
        if aug_idxes is None or len(aug_idxes) == 0:
            return data

        results = []
        for aug_idx in aug_idxes:
            original_word = tokens[aug_idx]
            print("original word: ",original_word)
            tokens[aug_idx] = 50264
            masked_text = ' '.join(tokens)
            print("BERT masked input: ",masked_text)
            #TODO: 按照模型的输入修改这里的输入，使能找到被mask位置的n个candidates。
            candidates = self.model(masked_text, target_word=original_word, n=n)

            results.append([candidate[0] for candidate in candidates])
        # results = []
        # for src, dest in zip(data.split(' '), tokens):
        #     results.append(self.align_capitalization(src, dest))
        return results, aug_idxes

    @classmethod
    def get_model(cls, config, model_path, device='cuda', force_reload=False, temperature=1.0, top_k=None, top_p=0):
        if 'bart' in model_path:
            return init_bart_model(config)

        raise ValueError('Model name value is unexpected. Only support bert and xlnet models.')

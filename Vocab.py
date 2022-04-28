from transformers import BartTokenizer, BartModel
import os,regex
import pickle
import torch
from LossAttack.utils.corpus import Corpus
from collections import Counter

class Vocab(object):
    def __init__(self, words, chars, tags, rels):
        self.rels = sorted(rels)
        self.rel_dict = {rel: i for i, rel in enumerate(self.rels)}
        self.n_rels = len(self.rels)
        self.barttkn = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir='./data/pretrained/bart-base')
        self.pad_index = self.barttkn.pad_token_id
        self.unk_index = self.barttkn.unk_token_id
        
    @classmethod
    def from_corpus(cls, corpus, min_freq=1):
        words = Counter(word.lower() for seq in corpus.words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        chars = list({char for seq in corpus.words for char in ''.join(seq)})
        rels = list({rel for seq in corpus.rels for rel in seq})
        tags = list({tag for seq in corpus.tags for tag in seq})
        vocab = cls(words, chars, tags, rels)

        return vocab
# -*- coding: utf-8 -*-
from LossAttack.libs.luna import fetch_best_ckpt_name
from LossAttack.utils.parser_helper import load_parser
from LossAttack.utils.corpus import Corpus
from LossAttack.models import PosTagger
from LossAttack.utils.data import TextDataset, batchify
from LossAttack.task import ParserTask

import torch


class Evaluate(object):

    def __call__(self, config):
        print("Load the models")
        vocab = torch.load(config.vocab)
        parser = load_parser(fetch_best_ckpt_name(config.parser_model))
        task = ParserTask(vocab, parser)
        if config.pred_tag:
            tagger = PosTagger.load(fetch_best_ckpt_name(config.tagger_model))
        else:
            tagger = None

        print("Load the dataset")
        corpus = Corpus.load(config.fdata)
        dataset = TextDataset(vocab.numericalize(corpus))
        # set the data loader
        loader = batchify(dataset, config.batch_size, config.buckets)

        print("Evaluate the dataset")
        loss, metric = task.evaluate(loader, config.punct, tagger, True)
        print(f"Loss: {loss:.4f} {metric}")

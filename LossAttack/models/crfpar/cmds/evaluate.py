# -*- coding: utf-8 -*-

from datetime import datetime
from LossAttack.models.crfpar import Model
from LossAttack.models.crfpar.cmds.cmd import CMD
from LossAttack.models.crfpar.utils.corpus import Corpus
from LossAttack.models.crfpar.utils.data import TextDataset, batchify


class Evaluate(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Evaluate the specified model and dataset.'
        )
        subparser.add_argument('--punct', action='store_true',
                               help='whether to include punctuation')
        subparser.add_argument('--proj', action='store_true',
                               help='whether to projectivise the outputs')
        subparser.add_argument('--fdata', default='data/ptb/test.conllx',
                               help='path to dataset')

        return subparser

    def __call__(self, args):
        super(Evaluate, self).__call__(args)

        print("Load the dataset")
        corpus = Corpus.load(args.fdata, self.fields)
        dataset = TextDataset(corpus, self.fields, args.buckets)
        # set the data loader
        dataset.loader = batchify(dataset, args.batch_size)
        print(f"{len(dataset)} sentences, "
              f"{len(dataset.loader)} batches, "
              f"{len(dataset.buckets)} buckets")

        print("Load the model")
        self.model = Model.load(args.model)
        print(f"{self.model}\n")

        print("Evaluate the dataset")
        start = datetime.now()
        loss, metric = self.evaluate(dataset.loader)
        total_time = datetime.now() - start
        print(f"Loss: {loss:.4f} {metric}")
        print(f"{total_time}s elapsed, "
              f"{len(dataset) / total_time.total_seconds():.2f} Sents/s")

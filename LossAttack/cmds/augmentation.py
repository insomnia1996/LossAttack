import logging
# logging.basicConfig(level=logging.INFO,filename='log/augmentation.log')
from LossAttack.utils.corpus import Corpus,init_sentence
import numpy as np
import torch
from LossAttack.models.tagger import PosTagger
from LossAttack.libs.luna.ckpt_utils import fetch_best_ckpt_name
from LossAttack.cmds.blackbox.blackboxmethod import CharTypo,InsertingPunct,Substituting,DeletingPunct
from LossAttack.utils.data import TextDataset,DataLoader,collate_fn
from LossAttack.cmds.attack import Attack
from LossAttack.utils.parser_helper import load_parser
from LossAttack.task import ParserTask

# for training data Augmentation
class Augmentation(Attack):
    def get_attack_seq_generator(self, config):
        method = config.blackbox_method
        input_type = config.input
        if input_type == 'char':
            return CharTypo(config, self.vocab)
        else:
            if method == 'insert':
                return InsertingPunct(config, self.vocab)
            elif method == 'substitute':
                return Substituting(config, self.task, self.vocab)
            elif method == 'delete':
                return DeletingPunct(config, self.vocab)

    def __call__(self, config):
        self.vocab = torch.load(config.vocab)
        self.parser = load_parser(fetch_best_ckpt_name(config.parser_model))
        self.task = ParserTask(self.vocab, self.parser)
        # load training data
        corpus = Corpus.load(config.ftrain)
        dataset = TextDataset(self.vocab.numericalize(corpus, training=True))
        loader = DataLoader(dataset=dataset, collate_fn=collate_fn)
        augmentation_corpus = Corpus([])
        training_data_number = len(corpus.sentences)
        self.attack_seq_generator = self.get_attack_seq_generator(config)

        # random prob to decide whether to change a specific training data
        # if prob[index] < augmentation_rate, augmented.
        prob = np.random.uniform(0.0, 1.0, size=(training_data_number,))
        for index, (seq_idx, tag_idx, chars, arcs, rels) in enumerate(loader):
            sentence = corpus.sentences[index]
            augmentation_corpus.sentences.append(sentence)
            if index % 1000 == 0:
                print("{} sentences have processed! ".format(index))

            if prob[index] < config.augmentation_rate:
                seqs = self.get_seqs_name(seq_idx)
                tags = self.get_tags_name(tag_idx)
                mask = self.get_mask(seq_idx, self.vocab.pad_index, punct_list=self.vocab.puncts)
                raw_metric = self.task.evaluate([(seq_idx, tag_idx, chars, arcs, rels)],mst=config.mst)
                augmentation_seq, _,  _,  _, _, _ = self.attack_seq_generator.generate_attack_seq(' '.join(seqs[1:]), seq_idx, tags, tag_idx, chars, arcs, rels, mask, raw_metric)
                augmentation_corpus.sentences.append(init_sentence(sentence.FORM, tuple(augmentation_seq[1:]), sentence.POS, sentence.HEAD, sentence.DEPREL))

        if config.input == 'char':
            saved_file = '{}/ptb_train_typo_only_substitute_{:.0f}%_{}.sd'.format(config.augmentation_dir, config.augmentation_rate*100, config.revised_rate)
        else:
            saved_file = '{}/ptb_train_{:.0f}%_{}.sd'.format(config.augmentation_dir, config.augmentation_rate*100, config.revised_rate)

        print("Complete! {} sentences have processed!".format(training_data_number))
        print("Current training data number is {}.".format(len(augmentation_corpus.sentences)))
        print("The augmentation data are saved to file {}".format(saved_file))
        augmentation_corpus.save(saved_file)

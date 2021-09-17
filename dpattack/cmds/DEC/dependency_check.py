# coding: utf-8
from datetime import datetime, timedelta
from dpattack.utils.corpus import Corpus,init_sentence
from dpattack.utils.vocab import Vocab, Vocab_for_DEC
from dpattack.utils.pretrained import Pretrained
from dpattack.utils.metric import ParserMetric as Metric
from dpattack.libs.luna.pytorch import cast_list
from dpattack.utils.parser_helper import is_chars_judger
from dpattack.utils.data import TextDataset, batchify
from dpattack.utils.metric import ParserMetric, TaggerMetric
from dpattack.task import DECTask
from dpattack.utils.parser_helper import init_model
from train import main_for_bi_tir

import argparse
from config import Config
import random,re,os
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR


# Dependency error check class
#TODO: 把这部分encoder架构想好，DEC是训练的trainer，不是模型架构。
class DEC(object):

    def fetch_data(self, number_path, sent_path, adv_path):
        print("Preprocess the data")
        with open(number_path,'r') as f:
            line = f.readline().strip()#只有一行，存了攻击成功的index
            success_idx = line.split(" ")
        sent_corpus = Corpus.load_special(sent_path, success_idx)
        adv_corpus = Corpus.load_special(adv_path, success_idx)
        
        #distinguish train or dev
        shuffle = 'train' in sent_path
        if shuffle:
            train_or_dev = 'train'
        else:
            train_or_dev='dev'
        with open('/data/luoyt/dpattack/data/ptb/synt_vocab.pkl', 'rb') as f:
            synt_vocab = pickle.load(f)#id dict of synt tree tokens
            '''
            {'<s>': 0, '<pad>': 1, '</s>': 2, '(': 3, 'ROOT': 4, 'S': 5, 'ADVP': 6, 'RB': 7, ')': 8, ',': 9, 'INTJ': 10, 'UH': 11, 'FW': 12, 'VP': 13, 'VBP': 14, 'ADJP': 15, 'JJ': 16, ':': 17, 'NP': 18, 'NN': 19, '.': 20, 'PRP': 21, 'VBD': 22,
            'VBG': 23, 'TO': 24, 'VB': 25, 'PRT': 26, 'FRAG': 27, 'SBAR': 28, 'IN': 29, 'QP': 30, 'VBZ': 31, 'CD': 32, 'PP': 33, 'VBN': 34, 'DT': 35, 'CC': 36, 'NNS': 37, 'PRP$': 38, 'WHNP': 39, 'WP': 40, 'LS': 41, 'NNP': 42, 'SINV': 43, 'PRN': 44,
            '``': 45, "''": 46, 'JJR': 47, 'WDT': 48, 'POS': 49, 'MD': 50, 'SQ': 51, 'SBARQ': 52, 'WHADVP': 53, 'WRB': 54, 'RP': 55, 'EX': 56, 'JJS': 57, 'X': 58, 'LST': 59, '-LRB-': 60, '-RRB-': 61, 'RBS': 62, 'UCP': 63, 'RBR': 64, 'WHPP': 65, 
            'PDT': 66, 'WHADJP': 67, 'NX': 68, 'CONJP': 69, '$': 70, 'WP$': 71, '#': 72, 'SYM': 73, 'NNPS': 74, 'RRC': 75, 'NAC': 76}
            '''
        
        if os.path.exists(self.config.vocab_dec):
            with open(self.config.vocab_dec, 'rb') as f:
                vocab = pickle.load(f)
        else:
            vocab = Vocab_for_DEC.from_corpus(corpus=sent_corpus, min_freq=2)
            vocab.read_embeddings(Pretrained.load(self.config.fembed, self.config.unk))
            with open(self.config.vocab_dec, 'wb') as f:
                pickle.dump(vocab, f)
        if not os.listdir(os.path.join(self.config.tensor_dir, train_or_dev)):
            ori_words, ori_tags, _, ori_arcs, ori_rels, ori_tree, ori_sub_len = vocab.numericalize(sent_corpus, synt_vocab)
            atk_words, atk_tags, _, atk_arcs, atk_rels, atk_tree, atk_sub_len = vocab.numericalize(adv_corpus, synt_vocab)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "ori_words.pkl"), 'wb') as f:
                pickle.dump(ori_words,f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "ori_tags.pkl"),'wb') as f:
                pickle.dump(ori_tags,f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "ori_arcs.pkl"),'wb') as f:
                pickle.dump(ori_arcs,f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "ori_rels.pkl"),'wb') as f:
                pickle.dump(ori_rels,f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "ori_sublen.pkl"),'wb') as f:
                pickle.dump(ori_sub_len,f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "ori_tree.pkl"), 'wb') as f:
                pickle.dump(ori_tree,f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "atk_words.pkl"),'wb') as f:
                pickle.dump(atk_words,f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "atk_tags.pkl"),'wb') as f:
                pickle.dump(atk_tags,f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "atk_arcs.pkl"),'wb') as f:
                pickle.dump(atk_arcs,f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "atk_rels.pkl"),'wb') as f:
                pickle.dump(atk_rels,f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "atk_sublen.pkl"),'wb') as f:
                pickle.dump(atk_sub_len,f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "atk_tree.pkl"), 'wb') as f:
                pickle.dump(atk_tree,f)
            print("Data saved...")
        else:
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "ori_words.pkl"),'rb') as f:
                ori_words = pickle.load(f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "ori_tags.pkl"),'rb') as f:
                ori_tags = pickle.load(f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "ori_arcs.pkl"),'rb') as f:
                ori_arcs = pickle.load(f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "ori_rels.pkl"),'rb') as f:
                ori_rels = pickle.load(f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "ori_sublen.pkl"),'rb') as f:
                ori_sub_len = pickle.load(f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "ori_tree.pkl"),'rb') as f:
                ori_tree = pickle.load(f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "atk_words.pkl"),'rb') as f:
                atk_words = pickle.load(f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "atk_tags.pkl"),'rb') as f:
                atk_tags = pickle.load(f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "atk_arcs.pkl"),'rb') as f:
                atk_arcs = pickle.load(f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "atk_rels.pkl"),'rb') as f:
                atk_rels = pickle.load(f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "atk_sublen.pkl"),'rb') as f:
                atk_sub_len = pickle.load(f)
            with open(os.path.join(self.config.tensor_dir, train_or_dev, "atk_tree.pkl"),'rb') as f:
                atk_tree = pickle.load(f)
            print("Data loaded...")
        print("sample: ",vocab.id2word(ori_words[0]),vocab.id2word(ori_words[1]),vocab.id2word(ori_words[2]))
        textset = TextDataset((ori_words, ori_tags, ori_arcs, ori_rels, ori_tree, ori_sub_len, 
                               atk_words, atk_tags, atk_arcs, atk_rels, atk_tree, atk_sub_len))

        #textset.check_correction()
        #TextDataset: (ori_words, ori_tags, ori_arcs, ori_rels, atk_words, atk_tags, atk_arcs, atk_rels)元组
        

        loader = batchify(dataset=textset,
                                batch_size=self.config.batch_size,
                                n_buckets=self.config.buckets,
                                shuffle=shuffle)
        
        if shuffle:
            self.config.update({
                'max_synt_len': textset.max_syntlen(),
                'max_length':textset.maxlen(),
                'n_train_words': vocab.n_train_words,
                'n_tags': vocab.n_tags,
                'n_rels': vocab.n_rels,
                'n_chars': vocab.n_chars,
                'pad_index': vocab.pad_index,
                'unk_index': vocab.unk_index
        })
        
            print(f"{'train:':6} {len(textset):5} sentences in total, "
              f"{len(loader):3} batches provided")

        else:
            if textset.maxlen()>self.config.max_length:
                self.config.update({
                'max_length':textset.maxlen()})
            if textset.max_syntlen()>self.config.max_synt_len:
                self.config.update({
                'max_synt_len': textset.max_syntlen()})
            print(f"{'dev:':6} {len(textset):5} sentences in total, "
              f"{len(loader):3} batches provided")
        print(self.config)
        return loader, vocab

    def __call__(self, config):#黑盒攻击调用函数
        main_for_bi_tir()
        '''
        self.config = config
        
        random.seed(config.seed)
        np.random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        train_loader,vocab = self.fetch_data(os.path.join(config.result_path, "black_substitute_train_0.01.txt"),
                        config.ftrain,
                        os.path.join(config.result_path, "black_substitute_train_0.01.conllx"),
                        )
        dev_loader, _ = self.fetch_data(os.path.join(config.result_path, "black_substitute_valid_0.01.txt"),
                        config.fdev,
                        os.path.join(config.result_path, "black_substitute_valid_0.01.conllx"),
                        )
        model = init_model(self.config)

        task = DECTask(vocab, model)
        best_e, best_metric = 1, ParserMetric()

        if torch.cuda.is_available():
            model = model.cuda()
        print(f"{model}\n")

        total_time = timedelta()
        task.optimizer = Adam(
            task.model.parameters(),
            config.lr,
            (config.beta_1, config.beta_2),
            config.epsilon
        )
        task.scheduler = ExponentialLR(
            task.optimizer,
            config.decay ** (1 / config.steps)
        )

        for epoch in range(1, config.epochs + 1):
            start = datetime.now()
            # train one epoch and update the parameters
            task.train(train_loader)
            #TODO
            print(f"Epoch {epoch} / {config.epochs}:")
            loss, train_metric = task.train(train_loader)
            print(f"{'train:':6} Loss: {loss:.4f} {train_metric}")
            loss, dev_metric = task.evaluate(dev_loader, config.punct)
            print(f"{'dev:':6} Loss: {loss:.4f} {dev_metric}")
            
            t = datetime.now() - start

            if dev_metric > best_metric and epoch > config.patience:
                best_e, best_metric = epoch, dev_metric
                
                task.model.save(config.dec_model + f".{best_e}")
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= config.patience:
                break

        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")'''

    


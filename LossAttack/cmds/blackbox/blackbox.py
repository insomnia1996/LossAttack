# coding: utf-8
from LossAttack.cmds.attack import Attack
from LossAttack.cmds.blackbox.blackboxmethod import Denoising, Substituting, CharTypo, InsertingPunct, DeletingPunct
from LossAttack.utils.corpus import Corpus,init_sentence
from LossAttack.utils.metric import ParserMetric as Metric
from LossAttack.libs.luna.pytorch import cast_list
from LossAttack.utils.parser_helper import is_chars_judger
from LossAttack.utils.parser_helper import load_parser
from LossAttack.task import ParserTask
from LossAttack.utils.metric import ParserMetric, TaggerMetric
from LossAttack.utils.data import TextDataset,collate_fn
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import os


# BlackBoxAttack class
class BlackBoxAttack(Attack):
    def __init__(self):
        super(BlackBoxAttack, self).__init__()
        

    def pre_attack(self, config):
        corpus,loader = super().pre_attack(config)
        self.parser.eval()
        ref_vocab = torch.load(config.vocab)
        ref_parser = load_parser(os.path.join(config.workspace, "saved_models","crf_par", "parser.best"))
        self.ref_task = ParserTask(ref_vocab, ref_parser)
        return corpus, loader

    def __call__(self, config):#黑盒攻击调用函数
        random.seed(1)
        np.random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

        corpus, loader = self.pre_attack(config)
        ftrain = Corpus.load(config.ftrain)
        vocab = torch.load(config.vocab)
        config.update({
            'max_length':ftrain.maxlen(),
            'n_words': vocab.n_train_words,
            'n_tags': vocab.n_tags,
            'n_rels': vocab.n_rels,
            'n_chars': vocab.n_chars,
            'pad_index': vocab.pad_index,
            'unk_index': vocab.unk_index
        })
        # for saving
        attack_corpus = Corpus([])

        # attack seq generator
        self.attack_seq_generator = self.get_attack_seq_generator(config)
        self.attack(loader, config, attack_corpus)

        # save to file
        if config.save_result_to_file:
            attack_corpus_save_path = self.get_attack_corpus_saving_path(config)
            attack_corpus.save(attack_corpus_save_path)
            print('Result after attacking has saved in {}'.format(attack_corpus_save_path))

    def get_attack_corpus_saving_path(self, config):
        if config.input == 'char':
            attack_corpus_save_path = '{}/black_typo_{}_{}.conllx'.format(config.result_path,
                                                                          config.blackbox_pos_tag if config.blackbox_index == 'pos' else config.blackbox_index,
                                                                          config.revised_rate)
        else:
            if config.blackbox_method == 'substitute':
                attack_corpus_save_path = '{}/black_{}_{}_{}.conllx'.format(config.result_path,
                                                                            config.blackbox_method,
                                                                            config.blackbox_pos_tag if config.blackbox_index == 'pos' else config.blackbox_index,
                                                                            config.revised_rate)
            else:
                attack_corpus_save_path = '{}/black_{}_{}.conllx'.format(config.result_path,
                                                                        config.blackbox_method,
                                                                         config.revised_rate)
        return attack_corpus_save_path

    def get_attack_seq_generator(self, config):
        method = config.blackbox_method
        input_type = config.input
        if input_type == 'char':
            return CharTypo(config, self.vocab, parser=self.parser)
        else:
            if method == 'insert':
                return InsertingPunct(config, self.vocab)
            elif method == 'substitute':
                return Substituting(config, self.task, self.vocab, parser=self.parser)
            elif method == 'delete':
                return DeletingPunct(config, self.vocab)
            elif method == 'denoise':
                return Denoising(config, self.vocab, parser=self.parser)


    def attack_for_each_process(self, config, loader, attack_corpus):#循环attack batch里的每个句子
        revised_numbers = 0

        # three metric:
        # metric_before_attack: the metric before attacking(origin)
        # metric_after_attack: the metric after black box attacking
        raw_metric_all = Metric()
        attack_metric_all = Metric()
        success_numbers = 0
        real_suc_numbers = 0 
        success_idx=[]
        attacked_sents=[]
        for index, (seq_idx, tag_idx, chars, arcs, rels) in enumerate(loader):
            #print(seq_idx, tag_idx)#seq_idx是输入句子，tag_idx是每个词的CPOS词性标注#(1,seq_len)
            mask = self.get_mask(seq_idx, self.vocab.pad_index, punct_list=self.vocab.puncts)
            seqs = self.get_seqs_name(seq_idx)
            tags = self.get_tags_name(tag_idx)
            # attack for one sentence
            try:
                raw_metric, attack_metric, \
                attack_seq, attack_arc, attack_rel, \
                revised_number, is_valid = self.attack_for_each_sentence(config, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask, index)
                raw_metric_all += raw_metric
                attack_metric_all += attack_metric
                if attack_metric.uas < raw_metric.uas:
                    success_numbers += 1
                    if is_valid:
                        real_suc_numbers += 1
                        print("valid substitution!")
                        success_idx.append(str(index))
                if config.save_result_to_file:
                    # all result ignores the first token <ROOT>
                    attack_corpus.append(init_sentence(seqs[1:-1],
                                                attack_seq[1:-1],
                                                tags[1:-1],
                                                cast_list(arcs)[1:-1],
                                                self.vocab.id2rel(rels)[1:-1],
                                                attack_arc[:-1],
                                                attack_rel[:-1]))
                revised_numbers += revised_number
                attacked_sents.append(" ".join(attack_seq))
                print("Sentence: {}, Revised: {} Before: {} After: {} ".format(index + 1, revised_number, raw_metric, attack_metric))
                print("===========================================")
            except AssertionError:
               success_numbers += 1
               success_idx.append(str(index))
               print("Sentence: {}, Reconstruction failed, sentence length does not match.".format(index +1))
               print("===========================================")

        print("calculating perplexity...")
        ppl = self.attack_seq_generator.cal_ppl_bygpt2(attacked_sents)
        print("perplexity: ",ppl)
        print("calculating valid attack percentage...")
        print("rate:", real_suc_numbers/success_numbers)
            
        return raw_metric_all, attack_metric_all, revised_numbers, success_numbers, real_suc_numbers, success_idx

    def attack_for_each_sentence(self, config, seq, seq_idx, tag, tag_idx, chars, arcs, rels, mask, index):#attack单句
        '''
        :param seqs:
        :param seq_idx:
        :param tags:
        :param tag_idx:
        :param arcs:
        :param rels:
        :return:
        '''
        # seq length: ignore the first token (ROOT) of each sentence
        with torch.no_grad():
            # for metric before attacking
            raw_loss, raw_metric = self.task.evaluate([(seq_idx, tag_idx, chars, arcs, rels)],mst=config.mst)
            # score_arc_before_attack, score_rel_before_attack = self.parser.forward(seq_idx, is_chars_judger(self.parser, tag_idx, chars))

            # for metric after attacking
            # generate the attack sentence under attack_index
            if config.blackbox_method == 'denoise':
                attack_seq, attack_tag_idx, attack_mask, attack_gold_arc, attack_gold_rel, revised_number = self.attack_seq_generator.generate_attack_seq(seq, tag_idx, arcs, rels, mask, index)
            else:
                attack_seq, attack_tag_idx, attack_mask, attack_gold_arc, attack_gold_rel, revised_number = self.attack_seq_generator.generate_attack_seq(' '.join(seq[1:]), seq_idx, tag, tag_idx, chars, arcs, rels, mask, raw_metric)
            
            # get the attack seq idx and tag idx
            attack_seq_idx = self.vocab.word2id(attack_seq).unsqueeze(0)
            if torch.cuda.is_available():
                attack_seq_idx = attack_seq_idx.cuda()

            if is_chars_judger(self.parser):
                attack_chars = self.get_chars_idx_by_seq(attack_seq)
                attack_loss, attack_metric = self.task.evaluate([(attack_seq_idx, None, attack_chars, arcs, rels)], mst=config.mst)
                _, attack_arc, attack_rel = self.task.predict([(attack_seq_idx, attack_tag_idx, attack_chars)], mst=config.mst)
            else:
                attack_loss, attack_metric = self.task.evaluate([(attack_seq_idx, attack_tag_idx, None, attack_gold_arc, attack_gold_rel)], mst=config.mst)
                _, attack_arc, attack_rel = self.task.predict([(attack_seq_idx, attack_tag_idx, None)], mst=config.mst)
            
            #multiple positions replacement, check validity
            _, ref_metric = self.ref_task.evaluate([(seq_idx, tag_idx, chars, arcs, rels)], mst=config.mst)

            _, refadv_metric = self.ref_task.evaluate([(attack_seq_idx, attack_tag_idx, None, attack_gold_arc, attack_gold_rel)], mst=config.mst)
            if refadv_metric.uas == ref_metric.uas and refadv_metric.las == ref_metric.las:
                is_valid = True
            else:
                is_valid = False
            # if true, valid substitution or original sentence; otherwise, invalid substitution.
                
                
            return raw_metric, attack_metric, attack_seq, attack_arc[0], attack_rel[0], revised_number, is_valid

    def calc_baseline_metric_for_each_process(self, config, loader, attack_corpus):
        revised_numbers = 0

        print("Load the attacked dataset")
        baseline_corp = Corpus.load("/home/lyt/LossAttack/data/result/result_word_0.15@ptb_test_3.3.0.sd.adv")
        print("adv corpus length: %d" %len(baseline_corp))
        atk_words, atk_tags, atk_chars, atk_arcs, atk_rels = self.vocab.numericalize(baseline_corp, training=True)
        # three metric:
        # metric_before_attack: the metric before attacking(origin)
        # metric_after_attack: the metric after black box attacking
        raw_metric_all = Metric()
        attack_metric_all = Metric()
        success_numbers = 0
        real_suc_numbers = 0 
        success_idx=[]
        attacked_sents=[]
        
        for index, (seq_idx, tag_idx, chars, arcs, rels) in enumerate(loader):
            #print(seq_idx, tag_idx)#seq_idx是输入句子，tag_idx是每个词的CPOS词性标注#(1,seq_len)
            mask = self.get_mask(seq_idx, self.vocab.pad_index, punct_list=self.vocab.puncts)
            seqs = self.get_seqs_name(seq_idx)
            tags = self.get_tags_name(tag_idx)
            # attack for one sentence
            try:
                raw_metric, attack_metric, \
                attack_seq, revised_number, is_valid = self.compare_for_each_sentence(config, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask, index, atk_words, atk_tags, atk_chars, atk_arcs, atk_rels)
                raw_metric_all += raw_metric
                attack_metric_all += attack_metric
                if attack_metric.uas < raw_metric.uas:
                    success_numbers += 1
                    if is_valid:
                        real_suc_numbers += 1
                        print("valid substitution!")
                        success_idx.append(str(index))
                revised_numbers += revised_number
                attacked_sents.append(" ".join(attack_seq))
                print("Sentence: {}, Revised: {} Before: {} After: {} ".format(index + 1, revised_number, raw_metric, attack_metric))
                print("===========================================")
            except AssertionError:
               success_numbers += 1
               success_idx.append(str(index))
               print("Sentence: {}, Reconstruction failed, sentence length does not match.".format(index +1))
               print("===========================================")

        print("calculating perplexity...")
        ppl = self.attack_seq_generator.cal_ppl_bygpt2(attacked_sents)
        print("perplexity: ",ppl)
        print("calculating valid attack percentage...")
        print("rate:", real_suc_numbers/success_numbers)
            
        return raw_metric_all, attack_metric_all, revised_numbers, success_numbers, real_suc_numbers, success_idx
    
    def compare_for_each_sentence(self, config, seq, seq_idx, tag, tag_idx, chars, arcs, rels, mask, index, atk_words, atk_tags, atk_chars, atk_arcs, atk_rels):#attack单句
        '''
        :param seqs:
        :param seq_idx:
        :param tags:
        :param tag_idx:
        :param arcs:
        :param rels:
        :return:
        '''
        # seq length: ignore the first token (ROOT) of each sentence
        with torch.no_grad():
            # for metric before attacking
            raw_loss, raw_metric = self.task.evaluate([(seq_idx, tag_idx, chars, arcs, rels)],mst=config.mst)
            # score_arc_before_attack, score_rel_before_attack = self.parser.forward(seq_idx, is_chars_judger(self.parser, tag_idx, chars))


            attack_seq_idx = atk_words[index].unsqueeze(0).cuda()
            attack_tag_idx = tag_idx.cuda()
            attack_gold_arc = arcs.cuda()
            attack_gold_rel = rels.cuda()
            attack_arcs = atk_arcs[index].unsqueeze(0).cuda()
            attack_rels = atk_rels[index].unsqueeze(0).cuda()
            #print("atk and ori: ", attack_seq_idx, seq_idx)
            revised_number = torch.sum(torch.ne(attack_seq_idx, seq_idx)).item()
            
            # get the attack seq idx and tag idx
            attack_seq = self.vocab.id2word(attack_seq_idx.squeeze(0))
            print("sentence after attack: ", attack_seq)
            if torch.cuda.is_available():
                attack_seq_idx = attack_seq_idx.cuda()
            
            
            #multiple positions replacement, check validity
            _, ref_metric = self.ref_task.evaluate([(seq_idx, tag_idx, chars, arcs, rels)], mst=config.mst)

            _, refadv_metric = self.ref_task.evaluate([(attack_seq_idx, attack_tag_idx, None, attack_gold_arc, attack_gold_rel)], mst=config.mst)

            attack_metric = ParserMetric()
            mask = attack_seq_idx.ne(self.vocab.pad_index)
            mask[:, 0] = 0
            puncts = attack_seq_idx.new_tensor(self.vocab.puncts)
            mask &= attack_seq_idx.t().ne(puncts).all(-1)
            attack_arcs, attack_rels = attack_arcs[mask], attack_rels[mask]
            attack_gold_arc, attack_gold_rel = attack_gold_arc[mask], attack_gold_rel[mask]

            attack_metric(attack_arcs, attack_rels, attack_gold_arc, attack_gold_rel)

            if refadv_metric.uas == ref_metric.uas and refadv_metric.las == ref_metric.las:
                is_valid = True
            else:
                is_valid = False
            # if true, valid substitution or original sentence; otherwise, invalid substitution.
                
                
            return raw_metric, attack_metric, attack_seq, revised_number, is_valid


    def update_metric(self, metric, s_arc,s_rel, gold_arc, gold_rel):
        pred_arc, pred_rel = self.decode(s_arc, s_rel)
        metric(pred_arc, pred_rel, gold_arc, gold_rel)

    def attack(self, loader, config, attack_corpus):
        metric_before_attack, metric_after_attack, revised_numbers, success_numbers, real_suc_numbers, success_idx = self.attack_for_each_process(config, loader, attack_corpus)
        if config.blackbox_method == 'substitute':
                attack_corpus_save_path = '{}/black_{}_{}_{}.txt'.format(config.result_path,
                                                                            config.blackbox_method,
                                                                            config.blackbox_index if config.blackbox_index == 'unk' else config.blackbox_pos_tag,
                                                                            config.revised_rate)
        elif config.blackbox_method == 'denoise':
                attack_corpus_save_path = '{}/black_{}_{}_{}.txt'.format(config.result_path,
                                                                            config.blackbox_method,
                                                                            config.blackbox_index if config.blackbox_index == 'unk' else config.blackbox_pos_tag,
                                                                            config.revised_rate)
        
        with open(attack_corpus_save_path, 'w') as f:
            s = " ".join(success_idx)
            f.write(s+"\n")
        corpus_len=2416#len(attack_corpus)
        print("Before attacking: {}".format(metric_before_attack))
        print("Black box attack. Method: {}, Rate: {}, Modified:{:.2f}".format(config.blackbox_method,config.revised_rate,revised_numbers/corpus_len))
        print("After attacking: {}".format(metric_after_attack))
        print("UAS Drop Rate: {:.2f}%, LAS Drop Rate: {:.2f}%, success rate: {:.2f}%, real success rate:{:.2f}%".format((metric_before_attack.uas - metric_after_attack.uas) * 100, (metric_before_attack.las - metric_after_attack.las) * 100, success_numbers/corpus_len*100, real_suc_numbers/corpus_len*100))

    def get_chars_idx_by_seq(self, sentence):
        chars = self.vocab.char2id(sentence).unsqueeze(0)
        if torch.cuda.is_available():
            chars = chars.cuda()
        return chars
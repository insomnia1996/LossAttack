from nltk.util import print_string
from LossAttack.utils.utils import get_blackbox_augmentor
from LossAttack.utils.corpus import Corpus
from LossAttack.cmds.blackbox.attackindex import *
from LossAttack.libs.luna.pytorch import cast_list
from LossAttack.utils.aug import CharTypoAug
from LossAttack.utils.tag_tool import gen_tag_dict
from nltk import CRFTagger
import regex as re
from LossAttack.models.neuronlp2.models import StackPtrNet
import os
from LossAttack.utils.parser_helper import load_parser
from LossAttack.task import ParserTask
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class BlackBoxMethod(object):
    def __init__(self, vocab):
        self.vocab = vocab

        self.FALSE_TOKEN = -1

    def generate_attack_seq(self, *args, **kwargs):
        pass

    def check_pos_tag(self, *args, **kwargs):
        pass

    def update_mask_arc_rel(self, mask, arc, rel, revised_list):
        pass

    def copy_str_to_list(self, seq):
        if isinstance(seq, str):
            return seq.split()
        elif isinstance(seq, list):
            return seq

    def insert_token_to_str_list(self, seq, index, token):
        seq_list = seq
        if isinstance(seq, str):
            seq_list = seq.split()
        seq_list.insert(index, token)
        return seq_list

    def duplicate_sentence_with_candidate_replacement(self, seq, candidate, index):
        candidates_number = len(candidate)
        duplicate_list = self.duplicate_sentence(self.copy_str_to_list(seq), candidates_number)
        for i in range(candidates_number):
            duplicate_list[i][index] = candidate[i]
        return duplicate_list

    def duplicate_sentence(self, seq, times):
        return [seq.copy() for _ in range(times)]


class Substituting(BlackBoxMethod):
    def __init__(self, config, task, vocab=None, parser=None):
        super(Substituting, self).__init__(vocab)
        self.task = task
        ref_vocab = torch.load(config.vocab)
        ref_parser = load_parser(os.path.join(config.workspace, "saved_models","crf_par", "parser.best"))
        self.ref_task = ParserTask(ref_vocab, ref_parser)
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.parser = parser
        self.ppl_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=config.language_model_path)
        self.ppl_model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=config.language_model_path).to(self.device)
        self.index = self.get_index(config, vocab, parser)
        self.aug = get_blackbox_augmentor(config, config.blackbox_model, config.path, config.revised_rate, vocab=vocab, ftrain=config.ftrain)
        self.tag_dict = gen_tag_dict(Corpus.load(config.ftrain), vocab, 2, False)
        if self.config.blackbox_tagger == 'crf':
            self.crf_tagger = CRFTagger()
            self.crf_tagger.set_model_file(config.crf_tagger_path)
        #self.ref_parser1 = StackPtrNet(self.config, biaffine=True, pos=True, char=False)
        #self.ref_parser1.load_state_dict(torch.load("/home/lyt/LossAttack/data/saved_models/stackptr/network.pt"))
        #self.ref_parser1.to(self.device)
        self.ref_parser2 = load_parser(os.path.join(self.config.workspace, "saved_models","crf_par", "parser.best"))


    def get_index(self, config, vocab=None, parser=None):
        if config.mode == 'augmentation':
            return AttackIndexRandomGenerator(config)
        if config.blackbox_index == 'pos':
            return AttackIndexPosTag(config)
        else:
            if parser is None and vocab is None:
                print('unk replacement can not missing LossAttack and vocab')
                exit()
            return AttackIndexUnkReplacement(config, vocab=vocab, parser=parser)

    def generate_attack_seq(self, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask, raw_metric=None):
        # generate word index to be attacked
        attack_index = self.index.get_attack_index(self.copy_str_to_list(seqs), seq_idx, tags, tag_idx, chars, arcs, rels, mask)
        # 这一步还没有把要攻击的位置mask掉
        # generate word candidates to be attacked
        candidates, indexes = self.substituting(seqs, attack_index)
        # check candidates by pos_tagger
        candidates, indexes = self.check_pos_tag(seqs, tags, candidates, indexes)
        #print("candidates: ",candidates)
        attack_seq, revised_number = self.check_uas(seqs, tag_idx, arcs, rels, candidates, indexes, raw_metric)
        print("After attack:", [Corpus.ROOT] + attack_seq)
        return [Corpus.ROOT] + attack_seq, tag_idx, mask, arcs, rels, revised_number

    def generate_attack_seq_bak(self, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask, raw_metric=None):
        # generate word index to be attacked
        attack_index = self.index.get_attack_index_bak(self.copy_str_to_list(seqs), seq_idx, tags, tag_idx, chars, arcs, rels, mask)
        # 这一步还没有把要攻击的位置mask掉
        # generate word candidates to be attacked
        candidates, indexes = self.substituting(seqs, attack_index)
        # check candidates by pos_tagger
        candidates, indexes = self.check_pos_tag(seqs, tags, candidates, indexes)
        #print("candidates: ",candidates)
        attack_seq, revised_number = self.check_uas_bak(seqs, tag_idx, arcs, rels, candidates, indexes, raw_metric)
        print("After attack:", [Corpus.ROOT] + attack_seq)
        return [Corpus.ROOT] + attack_seq, tag_idx, mask, arcs, rels, revised_number

    def uas_las(self, pred_arcs, pred_rels, gold_arcs, gold_rels):
        arc_mask = pred_arcs.eq(gold_arcs)
        rel_mask = pred_rels.eq(gold_rels) & arc_mask

        total = len(arc_mask)
        correct_arcs = arc_mask.sum().item()
        correct_rels = rel_mask.sum().item()
        return correct_arcs / (total + 1e-10), correct_rels / (total + 1e-10)

    def cal_ppl_bygpt2(self, sents):
        len_sents= len(sents)
        ppl=0
        for sent in sents:
            inputs = torch.tensor(self.ppl_tokenizer.encode(sent)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.ppl_model(inputs, labels=inputs)
            loss, logits = outputs[:2]
            ppl += torch.exp(loss).item()
        return ppl/len_sents

    def check_consistency(self, seqs, attack_seqs, tags, heads, rels):#raw_metric 原句指标
        flag=False
        if type(seqs)==list:
            seqs = self.vocab.word2id([Corpus.ROOT] + seqs).unsqueeze(0)
            if torch.cuda.is_available():
                seqs = seqs.cuda()
        if type(attack_seqs)==list:
            attack_seqs = self.vocab.word2id([Corpus.ROOT] + attack_seqs).unsqueeze(0)
            if torch.cuda.is_available():
                attack_seqs = attack_seqs.cuda()
        _, ref_metric = self.ref_task.evaluate([(seqs, tags, None, heads, rels)])
        _, refadv_metric = self.ref_task.evaluate([(attack_seqs, tags, None, heads, rels)])
        if ref_metric.uas==refadv_metric.uas and ref_metric.las==refadv_metric.las:#make sure the reference parser sees no changes before and after attack 
            flag=True
        return flag
        

    def substituting(self, seq, index):
        try:
            # generate the attack sentence by index
            candidates, revised_indexes = self.aug.substitute(seq, aug_idxes=index,n=99)
        except Exception:
            try:
                # if error happens, generate the attack sentence by random
                candidates, revised_indexes = self.aug.substitute(seq)
            except Exception:
                candidates = None
                revised_indexes = []
        return candidates, revised_indexes

    def update_mask_arc_rel(self, mask, arc, rel, revised_list):
        return mask, arc, rel

    def check_pos_tag(self, seqs, tags, origin_candidates, indexes):#先选candidate，然后checkPOS标注，有可能会导致candidates为空
        tag_check_candidates = []
        tag_check_indexes = []
        if origin_candidates is None:
            return tag_check_candidates, tag_check_indexes
        for index, candidate in zip(indexes,origin_candidates):
            candidate = self.check_pos_tag_under_each_index(seqs, tags, candidate, index)
            if len(candidate) != 0:
                tag_check_indexes.append(index)
                tag_check_candidates.append(candidate)
        return tag_check_candidates, tag_check_indexes

    def check_pos_tag_under_each_index(self, seqs, tags, candidate, index):#仅部分词性的词汇才被攻击

        if self.config.blackbox_tagger == 'dict':
            if tags[index + 1] not in self.tag_dict:
                return []

            word_list_with_same_tag = self.tag_dict[tags[index+1]]
            tag_check_candidate = []
            for i, cand in enumerate(candidate):
                cand_idx = self.vocab.word_dict.get(cand.lower(), self.vocab.unk_index)
                if cand_idx != self.vocab.unk_index:
                    if cand_idx in word_list_with_same_tag:
                        tag_check_candidate.append(cand)
                        if len(tag_check_candidate) > self.config.blackbox_candidates:
                            break
            return tag_check_candidate
        elif self.config.blackbox_tagger == 'crf':
            tag_check_candidate = []
            sents = self.duplicate_sentence_with_candidate_replacement(seqs, candidate, index)
            word_tag_list = self.crf_tagger.tag_sents(sents)
            for count,word_tag in enumerate(word_tag_list):
                if word_tag[index][1] == tags[index + 1]:
                    tag_check_candidate.append(candidate[count])
                    if len(tag_check_candidate) > self.config.blackbox_candidates:
                        break
            return tag_check_candidate

    def check_uas(self, seqs, tag_idx, arcs, rels, candidates, indexes, raw_metric):
        final_attack_seq = self.copy_str_to_list(seqs)
        revised_number = 0
        for index, candidate in zip(indexes, candidates):#不同位置idx及该位置对应的候选词
            index_flag = self.check_uas_under_each_index(seqs, tag_idx, arcs, rels, candidate, index, raw_metric)
            if index_flag!=-1:
                final_attack_seq[index] = candidate[index_flag]
                revised_number += 1
        return final_attack_seq, revised_number
    
    def check_uas_bak(self, seqs, tag_idx, arcs, rels, candidates, indexes, raw_metric):
        final_attack_seq = self.copy_str_to_list(seqs)
        revised_number = 0
        for index, candidate in zip(indexes, candidates):#不同位置idx及该位置对应的候选词
            index_flag = self.check_uas_under_each_index_bak(seqs, tag_idx, arcs, rels, candidate, index, raw_metric)
            if index_flag!=-1:
                final_attack_seq[index] = candidate[index_flag]
                revised_number += 1
        return final_attack_seq, revised_number

    def words2tensor(self, words, device=None):
        seq_idx = [[self.vocab.word_dict.get(word.lower(), self.vocab.unk_index) for word in words]]
        return torch.tensor(seq_idx, device=device)

    def check_uas_under_each_index(self, seqs, tag_idx, arcs, rels, candidate, index, raw_metric):
        current_compare_uas = raw_metric.uas
        current_index = CONSTANT.FALSE_TOKEN
        for i, cand in enumerate(candidate):
            seq_idx = self.copy_str_to_list(seqs)
            attack_seqs = self.copy_str_to_list(seqs)
            attack_seqs[index] = cand
            flag = self.check_consistency(seq_idx, attack_seqs, tag_idx, arcs, rels)
            attack_metric = self.get_metric_by_seqs(attack_seqs, tag_idx, arcs, rels)
            if attack_metric.uas < current_compare_uas and flag:#某个位置的所有候选词全比较一遍，找UAS下降最多的候选词进行替换。
                print("match cand:%s" %attack_seqs)
                current_compare_uas = attack_metric.uas
                current_index = i
        if current_index == CONSTANT.FALSE_TOKEN:
            return -1#如果没有能使UAS下降的就不替换
        else:
            return current_index#仅返回一句话UAS下降最多的候选token
    
    def check_uas_under_each_index_bak(self, seqs, tag_idx, arcs, rels, candidate, index, raw_metric):
        current_compare_uas = raw_metric.uas
        current_index = CONSTANT.FALSE_TOKEN
        for i, cand in enumerate(candidate):
            attack_seqs = self.copy_str_to_list(seqs)
            attack_seqs[index] = cand
            attack_metric = self.get_metric_by_seqs(attack_seqs, tag_idx, arcs, rels)
            if attack_metric.uas < current_compare_uas:
                current_compare_uas = attack_metric.uas
                current_index = i
        if current_index == CONSTANT.FALSE_TOKEN:
            return -1#如果没有能使UAS下降的就不替换
        else:
            return current_index

    def get_metric_by_seqs(self, attack_seqs, tag_idx, arcs, rels):
        attack_seq_idx = self.vocab.word2id([Corpus.ROOT] + attack_seqs).unsqueeze(0)
        if torch.cuda.is_available():
            attack_seq_idx = attack_seq_idx.cuda()
        if is_chars_judger(self.task.model):
            attack_chars = self.get_chars_idx_by_seq(attack_seqs)
            _, attack_metric = self.task.evaluate([(attack_seq_idx, None, attack_chars, arcs, rels)], mst=self.config.mst)
        else:
            attack_tag_idx = tag_idx.clone()
            _, attack_metric = self.task.evaluate([(attack_seq_idx, attack_tag_idx, None, arcs, rels)],mst=self.config.mst)
        return attack_metric

    def get_chars_idx_by_seq(self, sentence):
        chars = self.vocab.char2id(sentence).unsqueeze(0)
        if torch.cuda.is_available():
            chars = chars.cuda()
        return chars

    # def check_pos_tag_under_each_index(self, seqs, tag_idx, candidate, index):
    #     if len(candidate) == 0:
    #         return self.FALSE_TOKEN
    #     attack_seq_list = self.duplicate_sentence_with_candidate_replacement(seqs, candidate, index)
    #     # change seq list to idx
    #     # add <ROOT> token to the first token of each setence
    #     attack_seq_idx = torch.cat([self.vocab.word2id([Corpus.ROOT] + s).unsqueeze(0) for s in attack_seq_list], dim=0)
    #     attack_tag_idx = self.tagger.decorator_forward(attack_seq_idx, self.ROOT_TAG)
    #     tag_equal_flag = torch.eq(attack_tag_idx[:, index + 1], tag_idx[0, index + 1])
    #     if torch.sum(tag_equal_flag) != 0:
    #         attack_succeed_index = tag_equal_flag.nonzero().squeeze(0)
    #         return attack_succeed_index[0].item()
    #     return self.FALSE_TOKEN


# class Inserting(BlackBoxMethod):
#     def __init__(self, config, vocab, tagger, ROOT_TAG):
#         super(Inserting, self).__init__(vocab)
#         self.tagger = tagger
#         self.ROOT_TAG = ROOT_TAG
#         self.index = AttackIndexInserting(config)
#         self.aug = get_blackbox_augmentor(config.blackbox_model, config.path, config.revised_rate, vocab=vocab,ftrain=config.ftrain)
#
#     def generate_attack_seq(self, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask):
#         # seq_idx, tag_idx, arcs, rels, mask = map(lambda x:x.squeeze(0) if len(x.shape)==2 else x,[seq_idx, tag_idx, arcs, rels, mask])
#         # generate word index to be attacked
#         gold_arcs = cast_list(arcs)
#         attack_index = self.index.get_attack_index(self.copy_str_to_list(seqs), seq_idx, tags, tag_idx, chars, arcs, mask)
#         # generate word candidates to be attacked
#         candidates, indexes = self.inserting(seqs, attack_index)
#         # check candidates by pos_tagger
#         attack_seq, attack_mask, attack_gold_arc, attack_gold_rel, revised_number = self.check_pos_tag(seqs,
#                                                                                                        tags,
#                                                                                                        cast_list(mask),
#                                                                                                        gold_arcs,
#                                                                                                        self.vocab.id2rel(rels),
#                                                                                                        candidates,
#                                                                                                        indexes)
#
#
#         attack_mask = torch.tensor(attack_mask,dtype=mask.dtype)
#         attack_gold_arc = torch.tensor(attack_gold_arc, dtype=arcs.dtype)
#         attack_gold_rel = self.vocab.rel2id(attack_gold_rel)
#         attack_mask, attack_gold_arc, attack_gold_rel = map(lambda x:x.unsqueeze(0) if len(x.shape)==1 else x, [attack_mask, attack_gold_arc, attack_gold_rel])
#         attack_mask, attack_gold_arc, attack_gold_rel = map(lambda x:x.cuda() if torch.cuda.is_available() else x, [attack_mask, attack_gold_arc, attack_gold_rel])
#
#         return [Corpus.ROOT] + attack_seq, attack_mask, attack_gold_arc, attack_gold_rel, revised_number
#
#     def inserting(self, seq, index):
#         try:
#             # generate the attack sentence by index
#             candidates, revised_indexes = self.aug.insert(seq, aug_idxes = index)
#         except Exception:
#             candidates = None
#             revised_indexes = []
#         return candidates, revised_indexes
#
#     def check_pos_tag(self, seqs, tags, mask, arcs, rels, candidates, indexes):
#         final_attack_seq = self.copy_str_to_list(seqs)
#         #final_attack_tag = tags.copy()
#         #revised_index_list = []
#         revised_number = 0
#         for count, index in enumerate(indexes):
#             succeed = self.check_pos_tag_under_each_index(seqs, tags, candidates[count], index)
#             if succeed != self.FALSE_TOKEN:
#                 final_attack_seq.insert(index, candidates[count][succeed])
#                 mask.insert(index + 1, 0)
#                 arcs = self.get_gold_arc_under_inserting(tags, arcs, index)
#                 rels = self.get_gold_rel_under_inserting(tags, rels, index)
#                 revised_number += 1
#
#         return final_attack_seq, mask, arcs, rels, revised_number
#
#     def check_pos_tag_under_each_index(self, seqs, tags, candidate, index):
#         seqs_insert_token = self.insert_token_to_str_list(seqs, index, self.vocab.UNK)
#         attack_seq_list = self.duplicate_sentence_with_candidate_replacement(seqs_insert_token, candidate, index)
#         # change seq list to idx
#         # add <ROOT> token to the first token of each setence
#         attack_seq_idx = torch.cat([self.vocab.word2id([Corpus.ROOT] + s).unsqueeze(0) for s in attack_seq_list], dim=0)
#         attack_tag_idx = self.tagger.decorator_forward(attack_seq_idx, self.ROOT_TAG, return_device='cpu')
#         gold_tag_idx = self.get_gold_tag_idx_under_inserting(tags, index)
#         tag_equal_flag = torch.eq(attack_tag_idx[:, index + 1], gold_tag_idx[index + 1])
#         if torch.sum(tag_equal_flag) != 0:
#             attack_succeed_index = tag_equal_flag.nonzero().squeeze(0)
#             return attack_succeed_index[0].item()
#         return self.FALSE_TOKEN
#
#     def get_gold_tag_under_inserting(self, tag, index):
#         gold_tag = tag.copy()
#         if tag[index + 1] in CONSTANT.NOUN_TAG:
#             gold_tag.insert(index + 1, CONSTANT.ADJ_TAG)
#         elif tag[index].startswith(CONSTANT.VERB_TAG):
#             gold_tag.insert(index + 1, CONSTANT.ADV_TAG)
#         return gold_tag
#
#     def get_gold_arc_under_inserting(self, tag, arcs, index):
#         if tag[index + 1] in CONSTANT.NOUN_TAG:
#             arcs.insert(index + 1, index + 1)
#         elif tag[index].startswith(CONSTANT.VERB_TAG):
#             arcs.insert(index + 1, index)
#         return [arc + 1 if arc > index else arc for arc in arcs]
#
#
#     def get_gold_rel_under_inserting(self, tag, rel, index):
#         if tag[index + 1] in CONSTANT.NOUN_TAG:
#             rel.insert(index + 1, CONSTANT.JJ_REL_MODIFIER)
#         elif tag[index].startswith(CONSTANT.VERB_TAG):
#             rel.insert(index + 1, CONSTANT.RB_REL_MODIFIER)
#         return rel
#
#     def get_gold_tag_idx_under_inserting(self, tag, index):
#         gold_tag = self.get_gold_tag_under_inserting(tag, index)
#         return self.vocab.tag2id(gold_tag)
#
#
#     def update_mask(self, mask, revised_list):
#         if len(revised_list)==0:
#             return mask
#         mask_in_list = cast_list(mask)
#         for revised in revised_list:
#             mask_in_list.insert(revised, 0)
#         return torch.tensor(mask_in_list,dtype=mask.dtype)
#
#     def update_arcs(self,arcs,tags,revised_list):
#         if len(revised_list)==0:
#             return arcs
#         arcs_in_list = cast_list(arcs)
#         for revised in revised_list:
#             arcs_in_list = [arcs + 1 if arcs > revised else arcs for arcs in arcs_in_list]
#             arcs_in_list.insert(revised+1 if tags[revised].startswith(CONSTANT.ADJ_TAG) else revised-1,revised)
#         return torch.tensor(arcs_in_list,dtype=arcs.dtype)
#
#     def update_rels(self,rels,tags,revised_list):
#         if len(revised_list)==0:
#             return rels
#         rels_in_list = self.vocab.id2rel(rels)
#         for revised in revised_list:
#             rels_in_list.insert(CONSTANT.JJ_REL_MODIFIER if tags[revised].startswith(CONSTANT.ADJ_TAG) else CONSTANT.RB_REL_MODIFIER,revised)
#         return self.vocab.rel2id(rels_in_list)


class InsertingPunct(BlackBoxMethod):
    def __init__(self, config, vocab):
        super(InsertingPunct, self).__init__(vocab)

        self.index = AttackIndexInsertingPunct(config, vocab)
        self.COMMA_TAG_IDX = vocab.tag_dict[CONSTANT.COMMA]

    def generate_attack_seq(self, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask, raw_metric=None):
        attack_index = self.index.get_attack_index(self.copy_str_to_list(seqs), seq_idx, tags, tag_idx, chars, arcs, mask)

        attack_index.sort(reverse=True)
        attack_tags = cast_list(tag_idx)
        attack_mask = cast_list(mask)
        attack_arcs = cast_list(arcs)
        attack_rels = cast_list(rels)
        attack_seqs = self.copy_str_to_list(seqs)

        for index in attack_index:
            attack_seqs.insert(index, CONSTANT.COMMA)
            attack_tags.insert(index + 1, self.COMMA_TAG_IDX)
            attack_mask.insert(index + 1, 0)
            attack_rels.insert(index + 1, 0)
            attack_arcs.insert(index + 1, 0)
            attack_arcs = [arc + 1 if arc > index else arc for arc in attack_arcs]

        attack_tags = torch.tensor(attack_tags, dtype=tag_idx.dtype)
        attack_mask = torch.tensor(attack_mask, dtype=mask.dtype)
        attack_arcs = torch.tensor(attack_arcs, dtype=arcs.dtype)
        attack_rels = torch.tensor(attack_rels, dtype=rels.dtype)
        attack_tags, attack_mask, attack_arcs, attack_rels = map(lambda x: x.unsqueeze(0) if len(x.shape) == 1 else x,
                                                            [attack_tags, attack_mask, attack_arcs, attack_rels])
        attack_tags, attack_mask, attack_arcs, attack_rels = map(lambda x: x.cuda() if torch.cuda.is_available() else x,
                                                            [attack_tags, attack_mask, attack_arcs, attack_rels])
        return [Corpus.ROOT] + attack_seqs, attack_tags, attack_mask, attack_arcs, attack_rels, len(attack_index)


# class Deleting(BlackBoxMethod):
#     def __init__(self, config, vocab):
#         super(Deleting, self).__init__(vocab)
#         self.index = AttackIndexDeleting(config)
#
#     def generate_attack_seq(self, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask):
#         seq_idx, tag_idx, arcs, rels, mask = map(lambda x:x.squeeze(0) if len(x.shape)==2 else x,[seq_idx, tag_idx, arcs, rels, mask])
#
#         gold_arcs = cast_list(arcs)
#         attack_index = self.index.get_attack_index(self.copy_str_to_list(seqs), seq_idx, tags, tag_idx, chars, arcs, mask)
#         attack_index.sort(reverse=True)
#
#         attack_seq = [Corpus.ROOT] + seqs.split()
#         attack_mask = cast_list(mask)
#         attack_gold_arc = gold_arcs.copy()
#         attack_gold_rel = cast_list(rels)
#
#         for index in attack_index:
#             del attack_seq[index]
#             del attack_mask[index]
#             del attack_gold_arc[index]
#             del attack_gold_rel[index]
#             attack_gold_arc = [arc - 1 if arc > index else arc for arc in attack_gold_arc]
#
#         attack_mask = torch.tensor(attack_mask,dtype=mask.dtype)
#         attack_gold_arc = torch.tensor(attack_gold_arc, dtype=arcs.dtype)
#         attack_gold_rel = torch.tensor(attack_gold_rel, dtype=rels.dtype)
#         attack_mask, attack_gold_arc, attack_gold_rel = map(lambda x:x.unsqueeze(0) if len(x.shape)==1 else x, [attack_mask, attack_gold_arc, attack_gold_rel])
#         attack_mask, attack_gold_arc, attack_gold_rel = map(lambda x:x.cuda() if torch.cuda.is_available() else x, [attack_mask, attack_gold_arc, attack_gold_rel])
#
#         return attack_seq, attack_mask, attack_gold_arc, attack_gold_rel, len(attack_index)


class DeletingPunct(BlackBoxMethod):
    def __init__(self, config, vocab):
        super(DeletingPunct, self).__init__(vocab)

        self.index = AttackIndexDeletingPunct(config, vocab)

    def generate_attack_seq(self, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask, raw_metric=None):
        gold_arcs = cast_list(arcs)
        attack_index = self.index.get_attack_index(self.copy_str_to_list(seqs), seq_idx, tags, tag_idx, chars, gold_arcs, mask)

        attack_index.sort(reverse=True)
        attack_mask = cast_list(mask)
        attack_tags = cast_list(tag_idx)
        attack_arcs = gold_arcs.copy()
        attack_rels = cast_list(rels)
        attack_seqs = self.copy_str_to_list(seqs)

        for index in attack_index:
            del attack_seqs[index]
            del attack_tags[index + 1]
            del attack_mask[index + 1]
            del attack_arcs[index + 1]
            del attack_rels[index + 1]
            attack_arcs = [arc - 1 if arc > index else arc for arc in attack_arcs]

        attack_tags = torch.tensor(attack_tags, dtype=tag_idx.dtype)
        attack_mask = torch.tensor(attack_mask, dtype=mask.dtype)
        attack_arcs = torch.tensor(attack_arcs, dtype=arcs.dtype)
        attack_rels = torch.tensor(attack_rels, dtype=rels.dtype)
        attack_tags, attack_mask, attack_arcs, attack_rels = map(lambda x: x.unsqueeze(0) if len(x.shape) == 1 else x,
                                                            [attack_tags, attack_mask, attack_arcs, attack_rels])
        attack_tags, attack_mask, attack_arcs, attack_rels = map(lambda x: x.cuda() if torch.cuda.is_available() else x,
                                                            [attack_tags, attack_mask, attack_arcs, attack_rels])
        return [Corpus.ROOT] + attack_seqs, attack_tags, attack_mask, attack_arcs, attack_rels, len(attack_index)


class CharTypo(BlackBoxMethod):
    def __init__(self, config, vocab, parser=None):
        super(CharTypo,self).__init__(vocab)
        self.index = self.get_index(config, vocab, parser)
        self.aug = CharTypoAug(vocab.char_dict)

    def get_index(self, config, vocab=None, parser=None):
        if config.mode == 'augmentation':
            return AttackIndexRandomGenerator(config)
        if config.blackbox_index == 'pos':
            return AttackIndexPosTag(config)
        else:
            if parser is None and vocab is None:
                print('unk replacement can not missing LossAttack and vocab')
                exit()
            return AttackIndexUnkReplacement(config, vocab=vocab, parser=parser)

    def generate_attack_seq(self, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask, raw_metric=None):
        origin_seqs = ' '.join([self.vocab.id2char(charid) for charid in chars[0,1:]])
        # generate word index to be attacked
        attack_index = self.index.get_attack_index(self.copy_str_to_list(origin_seqs), seq_idx, tags, tag_idx, chars, arcs, mask)

        attack_seq = self.aug.get_typos(origin_seqs, attack_index)

        return [Corpus.ROOT] + attack_seq, tag_idx, mask, arcs, rels, len(attack_index)

class Denoising(BlackBoxMethod):# validation of examples reconstructed by Bi-TIR
    def __init__(self, config, vocab, parser):
        super(Denoising,self).__init__(vocab)
        self.vocab = vocab
        self.parser = parser
        with open("/home/lyt/LossAttack/data/result/gold_sort.txt",'r') as f:
            self.gold = f.readlines()
        with open("/home/lyt/LossAttack/data/result/res_sort.txt",'r') as f:
            self.pred = f.readlines()
        with open("/home/lyt/LossAttack/data/result/goldarc_sort.txt",'r') as f:
            self.goldarc = f.readlines()
        with open("/home/lyt/LossAttack/data/result/goldrel_sort.txt",'r') as f:
            self.goldrel = f.readlines()


    def generate_attack_seq(self, seq_idx, tags, arcs, rels, mask, seq_number):
        # generate word index to be attacked
        
        denoised_sent, tags, mask, arcs, rels, num= self.denoise(self.copy_str_to_list(seq_idx), tags, arcs, rels, mask, seq_number)
        
        #print(denoised_sent, tags, arcs, rels)
        #print(len(denoised_sent), tags.size(-1), arcs.size(-1), rels.size(-1))
        return denoised_sent, tags, mask, arcs, rels, num
    
    def copy_str_to_list(self, seq):
        if isinstance(seq, str):
            return seq.strip().split()
        elif isinstance(seq, list):
            return seq
    
    def denoise(self, seqs, tags, arcs, rels, mask, seq_number):
        denoised_sent = self.pred[seq_number].strip().split()
        denoised_sent = ['<s>'] + denoised_sent + ['</s>']
        assert len(denoised_sent)==tags.size(-1), "%d is not equal to %d. sent: %s ; gold: %s ." %(len(denoised_sent),tags.size(-1), " ".join(denoised_sent), " ".join(seqs))

        return denoised_sent, tags, mask, arcs, rels, 0
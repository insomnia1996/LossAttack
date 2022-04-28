# coding: utf-8
'''
package for generate word indexes to be attacked in a sentence
For insert, check
For delete,
For substitute, two method: unk(replace each word to <unk>) and pos_tag
'''
import math
import torch
import torch.nn.functional as F
import random
import numpy as np
from LossAttack.utils.constant import CONSTANT
from LossAttack.utils.parser_helper import is_chars_judger
from LossAttack.libs.luna.pytorch import cast_list
from LossAttack.models.char import CharParser
from transformers import GPT2Tokenizer,GPT2LMHeadModel
from collections import defaultdict
from gensim.models import word2vec


class AttackIndex(object):
    def __init__(self, config):
        self.revised_rate = config.revised_rate
        self.config = config

    def get_attack_index(self, *args, **kwargs):
        pass

    def get_number(self, revised_rate, length):#至少修改一个位置
        number = math.floor(revised_rate * length)
        if number == 0:
            number = 1
        return number

    def get_random_index_by_length_rate(self, index, revised_rate, length):
        number = self.get_number(revised_rate, length)
        if len(index) <= number:
            return index
        else:
            return np.random.choice(index, number)

class AttackIndexRandomGenerator(AttackIndex):
    def __init__(self, config):
        super(AttackIndexRandomGenerator, self).__init__(config)

    def get_attack_index(self, seqs, seq_idx, tags, tag_idx, chars, arcs, mask):
        sentence_length = len(seqs)
        number = self.get_number(self.revised_rate, sentence_length)
        return self.get_random_index_to_be_attacked(tags, sentence_length, number)

    def get_random_index_to_be_attacked(self, tags, length, number):
        if self.config.input == 'char':
            word_index = list(range(length))
        else:
            word_index = [index - 1 for index, tag in enumerate(tags) if tag in CONSTANT.REAL_WORD_TAGS]
        if len(word_index) <= number:
            return word_index
        else:
            return np.random.choice(word_index, number, replace=False)

class AttackIndexInserting(AttackIndex):
    def __init__(self, config):
        super(AttackIndexInserting, self).__init__(config)

    def get_attack_index(self, seqs, seq_idx, tags, tag_idx, chars, arcs, mask):
        index = []
        length = len(tags)
        for i in range(length):
            if tags[i] in CONSTANT.NOUN_TAG:
                # current index is a NN, check the word before it
                if self.check_noun(tags, i):
                    index.append(i - 1)
            elif tags[i].startswith(CONSTANT.VERB_TAG):
                # current index is a VB, check whether this VB is modified by a RB
                if self.check_verb(seqs[i-1], tags, arcs, i):
                    index.append(i)
        index = list(set(index))
        return index
        #return self.get_random_index_by_length_rate(index, self.revised_rate, length)

    def check_noun(self, tags, i):
        if i == 0:
            return True
        else:
            tag_before_word_i = tags[i-1]
            if not tag_before_word_i.startswith(CONSTANT.NOUN_TAG[0]) and not tag_before_word_i.startswith(CONSTANT.ADJ_TAG):
                return True
            return False

    def check_verb(self, verb, tags, arcs,i):
        if verb in CONSTANT.AUXILIARY_VERB:
            return False
        for tag, arc in zip(tags, arcs):
            if tag.startswith(CONSTANT.ADV_TAG) and arc == i:
                return False
        return True


class AttackIndexDeleting(AttackIndex):
    def __init__(self, config):
        super(AttackIndexDeleting, self).__init__(config)

    def get_attack_index(self, seqs, seq_idx, tags, tag_idx, chars, arcs, mask):
        index = []
        length = len(tags)
        for i in range(length):
            if tags[i].startswith(CONSTANT.ADJ_TAG) or tags[i].startswith(CONSTANT.ADV_TAG):
                if self.check_modifier(arcs,i):
                    index.append(i)
        return index

    def check_modifier(self, arcs, index):
        for arc in arcs:
            if arc == index:
                return False
        return True


class AttackIndexUnkReplacement(AttackIndex):
    def __init__(self, config, vocab = None, parser = None):
        super(AttackIndexUnkReplacement, self).__init__(config)
        self.parser = parser#train阶段训练好的word_tag模型，将每个word对应它的parser tag
        self.vocab = vocab
        self.unk_chars = self.get_unk_chars_idx(self.vocab.UNK)

    def decode(self, s_arc, s_rel, mask, mst):
        from LossAttack.utils.alg import eisner
        if mst:
            pred_arcs = eisner(s_arc, mask)
        else:
            pred_arcs = s_arc.argmax(dim = -1)
        # pred_arcs = s_arc.argmax(dim=-1)
        # pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)
        pred_rels = s_rel.argmax(-1)
        pred_rels = pred_rels.gather(-1, pred_arcs.unsqueeze(-1)).squeeze(-1)

        return pred_arcs, pred_rels

    def criterion(self, prob_q, prob_p):#prob_p为真实分布，prob_q为预测分布。
        return F.kl_div(F.log_softmax(prob_q, dim=-1), F.softmax(prob_p, dim=-1), reduction='sum')


    def get_kl_div_bak(self, s_arc,gold_arcs, s_rel,gold_rels):
        kl=[]
        #s_rel = torch.mean(s_rel, dim=-2)  # [seq_len, seq_len, 46]

        #print(s_arc.shape,gold_arcs.shape)# torch.Size([seq_len, seq_len, seq_len+num_punct]) torch.Size([1, seq_len])
        #print(s_rel.shape, gold_rels.shape)# torch.Size([seq_len, seq_len, n_rels]) torch.Size([1, seq_len])
        
        
        heads = s_arc.argmax(dim=2)
        #logits_deprel, true_deprels = s_rel[:, 1:], gold_rels[:, 1:]  # exclude root
        logits_deprel, true_deprels = s_rel, gold_rels
        gather_index = heads.view(*heads.size(), 1, 1).expand(-1, -1, -1, logits_deprel.size(-1))
        logits_deprel = torch.gather(logits_deprel, dim=2, index=gather_index).contiguous().squeeze(2)
        

        #print("rel shape: ",logits_deprel.shape, gold_rels.shape)#[seq_len,seq_len,n_rels],[1,seq_len]

        for idx in range(s_arc.shape[0]):
            arc_kl = self.criterion(s_arc[idx], gold_arcs[idx])
            rel_kl = self.criterion(logits_deprel[idx], gold_rels[idx])
            kl.append(torch.log(arc_kl).item()+torch.log(rel_kl).item())#未mask前，整句话的loss都一样，因此mask后loss越大，说明mask前后loss差越大
        return kl

    def get_kl_div(self, s_arc, gold_arcs, s_rel, gold_rels):
        kl=[]
        
        for idx in range(s_arc.shape[0]):
            arc_kl = self.criterion(s_arc[idx], gold_arcs[0])
            #rel_kl = self.criterion(s_rel[idx], gold_rels[0])
            kl.append(arc_kl.item())#+rel_kl.item())
        return kl

    def get_attack_index(self, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask):#LossAttack method
        # `mask`: 除CLS SEP 标点符号以外都为true(punct=False)
        print("Using KLAttack method...")
        length = torch.sum(mask).item()
        
        index_to_be_replace = cast_list(mask.squeeze(0).nonzero())#取mask True对应index
        #print(index_to_be_replace)
        # for metric when change a word to <unk>
        # change each word to <unk> in turn, taking the worst case.
        # For a seq_index [<ROOT>   1   2   3   ,   5]
        # seq_idx_unk is
        #  [[<ROOT>    <unk>    2   3   ,   5]
        #   [<ROOT>    1    <unk>   3   ,   5]
        #   [<ROOT>    1    2   <unk>   ,   5]
        #   [<ROOT>    1    2   3   ,   <unk>]]
        
        before_arc, before_rel = self.parser.forward(seq_idx, tag_idx)
        heads = before_arc.argmax(dim=2)
        #logits_deprel, true_deprels = s_rel[:, 1:], gold_rels[:, 1:]  # exclude root
        logits_deprel, true_deprels = before_rel, rels
        gather_index = heads.view(*heads.size(), 1, 1).expand(-1, -1, -1, logits_deprel.size(-1))
        before_rel = torch.gather(logits_deprel, dim=2, index=gather_index).contiguous().squeeze(2)
        
        seq_idx_unk = self.generate_unk_seqs(seq_idx, length, index_to_be_replace)
        if is_chars_judger(self.parser):
            char_idx_unk = self.generate_unk_chars(chars, length, index_to_be_replace)
            score_arc, score_rel = self.parser.forward(seq_idx_unk, char_idx_unk)
        else:
            tag_idx_unk = self.generate_unk_tags(tag_idx, length)# repeat
            score_arc, score_rel = self.parser.forward(seq_idx_unk, tag_idx_unk)#比较换成UNK句子生成的句法树与真实句法树的区别来计算loss
        
        # heads = score_arc.argmax(dim=2)
        # logits_deprel, true_deprels = s_rel[:, 1:], gold_rels[:, 1:]  # exclude root
        # logits_deprel, true_deprels = score_rel, rels
        # gather_index = heads.view(*heads.size(), 1, 1).expand(-1, -1, -1, logits_deprel.size(-1))
        # score_rel = torch.gather(logits_deprel, dim=2, index=gather_index).contiguous().squeeze(2)
        
        # before_arc = torch.zeros_like(score_arc[0].unsqueeze(0))#[1,34,34]
        # before_arc =before_arc.scatter_(dim=-1, index=arcs.unsqueeze(-1), value=1)
        # before_rel = torch.zeros_like(score_rel[0].unsqueeze(0))#[1,34,34,47]
        # before_rel =before_rel.scatter_(dim=-1, index=rels.unsqueeze(-1), value=1)
        
        pred_arc = score_arc#[:,mask.squeeze(0)]
        pred_rel = score_rel
        
        
        non_equal_numbers = self.get_kl_div(pred_arc[:,mask.squeeze(0)],before_arc[:,mask.squeeze(0)], pred_rel[:,mask.squeeze(0)], before_rel[:,mask.squeeze(0)])#[seq_len]
        #print("===============\n",pred_arc.shape, pred_rel.shape, mask.shape)
        # torch.Size([25, 30, 30]) torch.Size([25, 30, 30, 47]) torch.Size([1, 30])
        #print("non_equal: ",non_equal_numbers)#[seq_len],选择loss最大的mask位置进行替换
        non_equal_dict = self.get_non_equal_dict(non_equal_numbers,index_to_be_replace,tags,is_char=isinstance(self.parser, CharParser))
        index_to_be_attacked = self.get_index_to_be_attacked(non_equal_dict,self.get_number(self.revised_rate, len(seqs)+1 if isinstance(self.parser, CharParser)else length))
        # non_equal不大的话index_to_be_attacked就是non_equal_numbers对应的index，否则随机采样numbers个
        
        #nums = self.get_number(self.revised_rate, len(seqs) if isinstance(self.parser, CharParser)else length)
        #index_to_be_attacked = self.top_k(non_equal_numbers, k=nums)效果不好
        
        print("index_to_be_attacked: ", index_to_be_attacked)#seqs[ind] will be attacked
        return index_to_be_attacked#如果attack多个index，则返回多个值。
        
    def get_attack_index_bak(self, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask):#LossAttack method
        # `mask`: 除CLS SEP 标点符号以外都为true(punct=False)
        print("Using dpattack method...")
        length = torch.sum(mask).item()
        index_to_be_replace = cast_list(mask.squeeze(0).nonzero())#取mask True对应index
        # for metric when change a word to <unk>
        # change each word to <unk> in turn, taking the worst case.
        # For a seq_index [<ROOT>   1   2   3   ,   5]
        # seq_idx_unk is
        #  [[<ROOT>    <unk>    2   3   ,   5]
        #   [<ROOT>    1    <unk>   3   ,   5]
        #   [<ROOT>    1    2   <unk>   ,   5]
        #   [<ROOT>    1    2   3   ,   <unk>]]
        seq_idx_unk = self.generate_unk_seqs(seq_idx, length, index_to_be_replace)
        if is_chars_judger(self.parser):
            char_idx_unk = self.generate_unk_chars(chars, length, index_to_be_replace)
            score_arc, score_rel = self.parser.forward(seq_idx_unk, char_idx_unk)
        else:
            tag_idx_unk = self.generate_unk_tags(tag_idx, length)# repeat
            score_arc, score_rel = self.parser.forward(seq_idx_unk, tag_idx_unk)
        pred_arc = score_arc.argmax(dim=-1)#[seq_len, seq_len+3], 每一种UNK替换后的句子标签输出，共seq_len个句子，seq_len+3个标签以序列输出。

        non_equal_numbers = self.calculate_non_equal_numbers(pred_arc[:,mask.squeeze(0)], arcs[mask])#[seq_len],i位置为第i句标错head词的数量。
        non_equal_dict = self.get_non_equal_dict(non_equal_numbers,index_to_be_replace,tags,is_char=isinstance(self.parser, CharParser))
        #print("non_equal_dict", non_equal_dict)
        index_to_be_attacked = self.get_index_to_be_attacked(non_equal_dict,self.get_number(self.revised_rate, len(seqs) if isinstance(self.parser, CharParser)else length))
        # non_equal不大的话index_to_be_attacked就是non_equal_numbers对应的index，否则随机采样numbers个
        print("index_to_be_attacked: ", index_to_be_attacked)#如果attack多个index，则返回多个值。
        return index_to_be_attacked
        # sorted_index = sorted(range(length), key=lambda k: non_equal_numbers[k], reverse=True)
        # if number is None:
        #     number = self.get_number(self.revised_rate, length)
        # if isinstance(self.parser, CharParser):
        #     return [index_to_be_replace[index] - 1 for index in sorted_index[:number]]
        # else:
        #     return self.get_index_to_be_attacked(sorted_index,tags,index_to_be_replace,number)


    def top_k(self, lst, k):
        sorted_idx = sorted(range(len(lst)), key=lambda x: lst[x], reverse=True)
        return sorted_idx[:k]

    def generate_unk_seqs(self, seq, length, index_to_be_replace):
        '''
        :param seq: seq_idx [<ROOT>   1   2   3   4   5], shape: [length + 1]
        :param length: sentence length
        :return:
        # for metric when change a word to <unk>
        # change each word to <unk> in turn, taking the worst case.
        # For a seq_index [<ROOT>   1   2   3   ,   5]
        # seq_idx_unk is
        #  [[<ROOT>    <unk>    2   3   ,   5]
        #   [<ROOT>    1    <unk>   3   ,   5]
        #   [<ROOT>    1    2   <unk>   ,   5]
        #   [<ROOT>    1    2   3   ,   <unk>]]
            shape: [length, length + 1]
        '''
        unk_seqs = seq.repeat(length, 1)
        for count, index in enumerate(index_to_be_replace):
            unk_seqs[count, index] = self.vocab.unk_index
        return unk_seqs

    def generate_unk_tags(self, tag, length):
        return tag.repeat(length, 1)

    def generate_unk_chars(self, char, length, index_to_be_replace):
        unk_chars = char.repeat(length, 1, 1)
        for count, index in enumerate(index_to_be_replace):
            unk_chars[count, index] = self.unk_chars
        return unk_chars

    def calculate_non_equal_numbers(self, pred_arc, gold_arc):
        '''
        :param pred_arc: pred arc 
        :param gold_arc: gold arc
        :return: the error numbers list
        '''
        non_equal_numbers = [torch.sum(torch.ne(arc, gold_arc)).item() for arc in pred_arc]
        return non_equal_numbers

    def get_unk_chars_idx(self, UNK_TOKEN):
        unk_chars = self.vocab.char2id([UNK_TOKEN]).squeeze(0)
        if torch.cuda.is_available():
            unk_chars = unk_chars.cuda()
        return unk_chars

    def get_non_equal_dict(self, non_equal_numbers, index_to_be_replace, tags, is_char=False):
        non_equal_dict = defaultdict(lambda: list())
        for index, non_equal_number in enumerate(non_equal_numbers):
            index_in_sentence = index_to_be_replace[index]
            if not is_char:
                if tags[index_in_sentence] in CONSTANT.REAL_WORD_TAGS:
                    non_equal_dict[non_equal_number].append(index_in_sentence - 1)
            else:
                non_equal_dict[non_equal_number].append(index_in_sentence - 1)
        return non_equal_dict#字典, k:v = 预测tag与真实不一样的次数:应修改第idx个词语为UNK(即non_equal_numbers中的值对应哪些index)
    
    def get_index_to_be_attacked(self, non_equal_dict, number):
        index_to_be_attacked = []
        current_number = number
        for key in sorted(non_equal_dict.keys(), reverse=True):#优先改错误数目/kldiv最大的位置，再试下一个位置。
            if len(non_equal_dict[key]) <= current_number:
                index_to_be_attacked.extend(non_equal_dict[key])
                current_number -= len(non_equal_dict[key])
            else:
                index_to_be_attacked.extend(np.random.choice(non_equal_dict[key], current_number, replace=False))
                current_number = 0
            if current_number == 0:
                break
        return index_to_be_attacked

class AttackIndexPosTag(AttackIndex):
    def __init__(self, config):
        super(AttackIndexPosTag, self).__init__(config)
        self.pos_tag = config.blackbox_pos_tag

    def get_attack_index(self, seqs, seq_idx, tags, tag_idx, chars, arcs, mask):
        index = [index - 1 for index, tag in enumerate(tags) if tag.startswith(self.pos_tag)]
        return self.get_random_index_by_length_rate(index, self.revised_rate, len(tags))


class AttackIndexInsertingPunct(AttackIndex):
    def __init__(self, config, vocab):
        super(AttackIndexInsertingPunct, self).__init__(config)

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.vocab = vocab
        self.puncts = self.vocab.id2word(self.vocab.puncts)

    def get_sentence_score(self, sentence):
        tokenize_input = self.tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([[self.tokenizer.eos_token_id] + self.tokenizer.convert_tokens_to_ids(tokenize_input)])
        if torch.cuda.is_available():
            tensor_input = tensor_input.cuda()
        output = self.model(tensor_input, labels=tensor_input)
        loss, logits = output[:2]
        #print("gen_sent score: ",self.tokenizer.decode(torch.argmax(logits[-1, :, :], -1)), -loss.item() * len(tokenize_input))
        return -loss.item() * len(tokenize_input)

    def get_attack_index(self, seqs, seq_idx, tags, tag_idx, chars, arcs, mask):
        comma_insert_index, comma_insert_seqs = self.duplicate_sentence_with_comma_insertion(seqs, len(seqs))
        #print(comma_insert_seqs)#每一个空位都添加一次逗号，形成句子列表
        if len(comma_insert_index) == 0:
            return []
        with torch.no_grad():
            seq_scores = [self.get_sentence_score(seq) for seq in comma_insert_seqs]
        sorted_index = sorted(range(len(seq_scores)), key=lambda k: seq_scores[k], reverse=True)#按照LAS(其实就是loss)分数从小到大排序句子中词语的index
        number = self.get_number(self.revised_rate, len(seqs))
        return [comma_insert_index[index] for index in sorted_index[:number]]

    def duplicate_sentence_with_comma_insertion(self, seqs, length):
        duplicate_seqs_list = []
        comma_insert_index = []
        for index in range(1, length):
            if seqs[index] not in self.puncts and seqs[index - 1] not in self.puncts:#puncts为标点符号
                duplicate_seqs = seqs.copy()
                duplicate_seqs.insert(index, CONSTANT.COMMA)
                duplicate_seqs_list.append(duplicate_seqs)
                comma_insert_index.append(index)
        return comma_insert_index, [' '.join(seq) for seq in duplicate_seqs_list]


class AttackIndexDeletingPunct(AttackIndex):
    def __init__(self, config, vocab):
        super(AttackIndexDeletingPunct, self).__init__(config)

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.vocab = vocab
        self.puncts = self.vocab.id2word(self.vocab.puncts)

    def get_sentence_score(self, sentence):
        tokenize_input = self.tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([[self.tokenizer.eos_token_id] + self.tokenizer.convert_tokens_to_ids(tokenize_input)])
        if torch.cuda.is_available():
            tensor_input = tensor_input.cuda()
        output = self.model(tensor_input, labels=tensor_input)
        loss, logits = output[:2]
        return -loss.item() * len(tokenize_input)

    def get_attack_index(self, seqs, seq_idx, tags, tag_idx, chars, arcs, mask):
        comma_insert_index, comma_insert_seqs = self.duplicate_sentence_with_comma_insertion(seqs, len(seqs), arcs)
        if len(comma_insert_index) == 0:
            return []
        with torch.no_grad():
            seq_scores = [self.get_sentence_score(seq) for seq in comma_insert_seqs]
        sorted_index = sorted(range(len(seq_scores)), key=lambda k: seq_scores[k], reverse=True)
        number = self.get_number(self.revised_rate, len(seqs))
        return [comma_insert_index[index] for index in sorted_index[:number]]

    def duplicate_sentence_with_comma_insertion(self, seqs, length, arcs):
        duplicate_seqs_list = []
        punct_delete_index = []
        for index in range(length - 1):
            if seqs[index] in self.puncts:
                if self.check_arcs(index, arcs):
                    duplicate_seqs = seqs.copy()
                    del duplicate_seqs[index]
                    duplicate_seqs_list.append(duplicate_seqs)
                    punct_delete_index.append(index)
        return punct_delete_index, [' '.join(seq) for seq in duplicate_seqs_list]

    def check_arcs(self, index, arcs):
        target_index = index + 1
        for arc in arcs:
            if target_index == arc:
                return False
        return True
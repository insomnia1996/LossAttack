# -*- coding: utf-8 -*-

from collections import Counter
from stanfordcorenlp import StanfordCoreNLP
import regex
import torch
from dpattack.libs.luna import cast_list
from transformers import BartTokenizer, BartModel
from tqdm import trange

class Vocab(object):
    PAD = '<PAD>'
    UNK = '<UNK>'

    def __init__(self, words, chars, tags, rels):
        self.pad_index = 0
        self.unk_index = 1

        self.words = [self.PAD, self.UNK] + sorted(words)
        self.chars = [self.PAD, self.UNK] + sorted(chars)
        self.tags = [self.PAD, self.UNK] + sorted(tags)
        self.rels = sorted(rels)
        
        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.tag_dict = {tag: i for i, tag in enumerate(self.tags)}
        self.rel_dict = {rel: i for i, rel in enumerate(self.rels)}
        self.char_dict = {char: i for i, char in enumerate(self.chars)}

        # ids of punctuation that appear in words
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if regex.match(r'\p{P}+$', word))

        self.n_words = len(self.words)
        self.n_chars = len(self.chars)
        self.n_tags = len(self.tags)
        self.n_rels = len(self.rels)
        self.n_train_words = self.n_words
        self.barttkn = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir='/data/luoyt/dpattack/data/pretrained/bart-base')
        

    def __repr__(self):
        info = f"{self.__class__.__name__}: "
        info += f"{self.n_words} words, "
        info += f"{self.n_tags} tags, "
        info += f"{self.n_rels} rels"

        return info

    def word2emb(self, sequence):
        #use barttokenizer to fit pretrained weights
        out_emb=[]
        for word in sequence:
            subwords = torch.tensor(self.barttkn(word.lower())['input_ids'][1:-1])#[n_subwords]
            embed = self.embeddings(subwords)
            #avg embedding
            embed = torch.mean(embed, dim=0, keepdim=True)
            out_emb.append(embed)
        out_emb = torch.cat(out_emb, dim=0)#[seq_len, embed_dim]
        return out_emb
    
    def word2id(self, sequence):
        
        return torch.tensor([self.word_dict.get(word.lower(), self.unk_index)
                             for word in sequence])

    def id2word(self, ids):
        #use barttokenizer to fit pretrained weights
        ids = cast_list(ids)
        return [self.words[idx] for idx in ids]


    def tag2id(self, sequence):
        return torch.tensor([self.tag_dict.get(tag, self.unk_index)
                             for tag in sequence])

    def id2tag(self, ids):
        ids = cast_list(ids)
        return [self.tags[i] for i in ids]

    def rel2id(self, sequence):
        return torch.tensor([self.rel_dict.get(rel, 0)
                             for rel in sequence])

    def id2rel(self, ids):
        ids = cast_list(ids)
        return [self.rels[i] for i in ids]

    def char2id(self, sequence, max_length=20):
        char_ids = torch.zeros(len(sequence), max_length, dtype=torch.long)
        for i, word in enumerate(sequence):
            ids = torch.tensor([self.char_dict.get(c, self.unk_index)
                                for c in word[:max_length]])
            char_ids[i, :len(ids)] = ids

        return char_ids

    def id2char(self, ids):
        ids = cast_list(ids)
        return ''.join([self.chars[i] for i in ids if i!=0])

    def read_embeddings(self, embed, smooth=True):
        # if the UNK token has existed in the pretrained,
        # then use it to replace the one in the vocab
        '''
        if embed.unk:
            self.UNK = embed.unk

        # self.extend(embed.tokens)
        self.embeddings = torch.zeros(self.n_words, embed.dim)

        for i, word in enumerate(self.words):
            if word in embed:
                self.embeddings[i] = embed[word]
        if smooth:
            self.embeddings /= torch.std(self.embeddings)
        '''
        bart = BartModel.from_pretrained('facebook/bart-base', cache_dir='/data/luoyt/dpattack/data/pretrained/bart-base')
        self.embeddings = bart.get_input_embeddings()
        
        #print("embedding shape: ", self.embeddings.weight.shape)
        self.n_words = self.embeddings.weight.size(0)




    def extend(self, words):
        self.words.extend(sorted(set(words).difference(self.word_dict)))
        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if regex.match(r'\p{P}+$', word))
        self.n_words = len(self.words)

    def numericalize(self, corpus, training=True):
        #use barttokenizer to fit pretrained weights word2emb replace word2id
        words = [self.word2id(seq) for seq in corpus.words]
        #embeds = [self.word2emb(seq) for seq in corpus.words]
        tags = [self.tag2id(seq) for seq in corpus.tags]
        chars = [self.char2id(seq) for seq in corpus.words]
        if not training:
            return words, tags, chars
        arcs = [torch.tensor(seq) for seq in corpus.heads]
        rels = [self.rel2id(seq) for seq in corpus.rels]
        return words, tags, chars, arcs, rels

    @classmethod
    def from_corpus(cls, corpus, min_freq=1):
        words = Counter(word.lower() for seq in corpus.words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        chars = list({char for seq in corpus.words for char in ''.join(seq)})
        rels = list({rel for seq in corpus.rels for rel in seq})
        tags = list({tag for seq in corpus.tags for tag in seq})
        vocab = cls(words, chars, tags, rels)

        return vocab



class Vocab_for_DEC(object):
    PAD = '<PAD>'
    UNK = '<UNK>'

    def __init__(self, words, chars, tags, rels):
        self.pad_index = 0
        self.unk_index = 1

        self.words = [self.PAD, self.UNK] + sorted(words)
        self.chars = [self.PAD, self.UNK] + sorted(chars)
        self.tags = [self.PAD, self.UNK] + sorted(tags)
        self.rels = sorted(rels)
        
        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.tag_dict = {tag: i for i, tag in enumerate(self.tags)}
        self.rel_dict = {rel: i for i, rel in enumerate(self.rels)}
        self.char_dict = {char: i for i, char in enumerate(self.chars)}

        # ids of punctuation that appear in words
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if regex.match(r'\p{P}+$', word))

        self.n_words = len(self.words)
        self.n_chars = len(self.chars)
        self.n_tags = len(self.tags)
        self.n_rels = len(self.rels)
        self.n_train_words = self.n_words
        self.barttkn = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir='/data/luoyt/dpattack/data/pretrained/bart-base')
        #self.nlp = StanfordCoreNLP(r'/data/luoyt/stanford-corenlp-4.2.2')不能放在init里，否则pickle有问题
        
        

    def __repr__(self):
        info = f"{self.__class__.__name__}: "
        info += f"{self.n_words} words, "
        info += f"{self.n_tags} tags, "
        info += f"{self.n_rels} rels"

        return info

    def is_paren(self, tok):
        return tok == ")" or tok == "("

    def deleaf(self, tree):
        nonleaves = ''
        for w in tree.replace('\n', '').split():
            w = w.replace('(', '( ').replace(')', ' )')
            nonleaves += w + ' '

        arr = nonleaves.split()
        for n, i in enumerate(arr):
            if n + 1 < len(arr):
                tok1 = arr[n]
                tok2 = arr[n + 1]
                if not self.is_paren(tok1) and not self.is_paren(tok2):
                    arr[n + 1] = ""

        nonleaves = " ".join(arr)
        return nonleaves.split()

    def word2emb(self, sequence):
        #use barttokenizer to fit pretrained weights
        out_emb=[]
        for word in sequence:
            subwords = torch.tensor(self.barttkn(word.lower())['input_ids'][1:-1])#[n_subwords]
            embed = self.embeddings(subwords)
            #avg embedding
            embed = torch.mean(embed, dim=0, keepdim=True)
            out_emb.append(embed)
        out_emb = torch.cat(out_emb, dim=0)#[seq_len, embed_dim]
        return out_emb

    def word2tree(self, sequence, synt_vocab, max_synt_len=160):
        seq_str = " ".join(sequence[1:-1])#已有BOS&EOS，去掉
        tree = ['<s>'] + self.deleaf(self.nlp.parse(seq_str)) + ['</s>']#把ROOT<s></s>从语法树解析中去除
        if len(tree)>max_synt_len+1:
            synt= torch.tensor([(synt_vocab[tag] if tag in synt_vocab else 1)  for tag in tree[:max_synt_len+1]])
        else:
            synt= torch.tensor([(synt_vocab[tag] if tag in synt_vocab else 1) for tag in tree])
        return synt

    def word2id(self, sequence):
        #改用barttokenizer进行word转id，添加sublen用于记录每个word的subword长度。
        subwords =[self.barttkn.convert_tokens_to_ids(word) for word in self.barttkn.tokenize_bak(" ".join(sequence))]
        sub_len = [len(subword) for subword in subwords]
        out_words=[]
        for subword in subwords:
            out_words+=subword
        
        assert len(sub_len)==len(sequence)
        assert sum(sub_len)==len(out_words)
        out_words = torch.tensor(out_words)
        return out_words, sub_len

    def id2word(self, ids):
        #use barttokenizer to fit pretrained weights
        ids = cast_list(ids)
        tokens = self.barttkn.convert_ids_to_tokens(ids)
        text = "".join(tokens).replace("Ġ"," ")
        return text


    def tag2id(self, sequence, sub_len):
        out_tags=[]
        #assert len(sub_len)==len(sequence)
        for idx in range(len(sub_len)):
            tag = [self.tag_dict.get(sequence[idx], self.unk_index)] * sub_len[idx]
            out_tags+=tag
        return torch.tensor(out_tags)

    def id2tag(self, ids, sub_len):
        ids = cast_list(ids)
        out_tags=[]
        cnt=0
        for idx in sub_len:
            tag = self.tags[ids[cnt]]
            out_tags.append(tag)
            cnt+=idx
        assert len(out_tags)==len(sub_len)
        return out_tags

    def rel2id(self, sequence, sub_len):
        out_rels=[]
        #assert len(sub_len)==len(sequence)
        for idx in range(len(sub_len)):
            rel = [self.rel_dict.get(sequence[idx], 0)] * sub_len[idx]
            out_rels+=rel
        return torch.tensor(out_rels)

    def id2rel(self, ids, sub_len):
        ids = cast_list(ids)
        out_rels=[]
        cnt=0
        for idx in sub_len:
            rel = self.rels[ids[cnt]]
            out_rels.append(rel)
            cnt+=idx
        assert len(out_rels)==len(sub_len)
        return out_rels

    def arc2id(self, sequence, sub_len):
        out_arcs=[]
        #assert len(sub_len)==len(sequence)
        for idx in range(len(sub_len)):
            arc = [sequence[idx]]* sub_len[idx]
            out_arcs += arc
        return torch.tensor(out_arcs)


    def char2id(self, sequence, max_length=20):
        char_ids = torch.zeros(len(sequence), max_length, dtype=torch.long)
        for i, word in enumerate(sequence):
            ids = torch.tensor([self.char_dict.get(c, self.unk_index)
                                for c in word[:max_length]])
            char_ids[i, :len(ids)] = ids

        return char_ids


    def id2char(self, ids):
        ids = cast_list(ids)
        return ''.join([self.chars[i] for i in ids if i!=0])

    def read_embeddings(self, embed, smooth=True):
        # if the UNK token has existed in the pretrained,
        # then use it to replace the one in the vocab
        '''
        if embed.unk:
            self.UNK = embed.unk

        # self.extend(embed.tokens)
        self.embeddings = torch.zeros(self.n_words, embed.dim)

        for i, word in enumerate(self.words):
            if word in embed:
                self.embeddings[i] = embed[word]
        if smooth:
            self.embeddings /= torch.std(self.embeddings)
        '''
        bart = BartModel.from_pretrained('facebook/bart-base', cache_dir='/data/luoyt/dpattack/data/pretrained/bart-base')
        self.embeddings = bart.get_input_embeddings()
        
        self.n_words = self.embeddings.weight.size(0)




    def extend(self, words):
        self.words.extend(sorted(set(words).difference(self.word_dict)))
        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if regex.match(r'\p{P}+$', word))
        self.n_words = len(self.words)

    def numericalize(self, corpus, synt_vocab, training=True):
        #use barttokenizer to fit pretrained weights word2emb replace word2id
        words=[]
        sub_len=[]
        tags=[]
        chars=[]
        arcs=[]
        rels=[]
        for idx in trange(len(corpus.words)):
            if idx%1000==0:
                print("numericalize %d sentences..." %idx)
            seq = corpus.words[idx]
            tmp1, tmp2 = self.word2id(seq)
            words.append(tmp1)
            sub_len.append(tmp2)
        self.nlp = StanfordCoreNLP(r'/data/luoyt/stanford-corenlp-4.0.0', memory='8g', timeout=50000)
        chars = [self.char2id(seq) for seq in corpus.words]
        tags = [self.tag2id(seq,sub_len[idx]) for idx, seq in enumerate(corpus.tags)]
        arcs = [self.arc2id(seq,sub_len[idx]) for  idx, seq in enumerate(corpus.heads)]
        rels = [self.rel2id(seq,sub_len[idx]) for idx, seq in enumerate(corpus.rels)]
        tree = [self.word2tree(seq, synt_vocab) for seq in corpus.words]

        print("Numericalize length : ",len(words),len(tags),len(chars),len(arcs),len(rels),len(tree),len(sub_len))
        return words, tags, chars, arcs, rels, tree, [torch.tensor(sub) for sub in sub_len]

    @classmethod
    def from_corpus(cls, corpus, min_freq=1):
        words = Counter(word.lower() for seq in corpus.words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        chars = list({char for seq in corpus.words for char in ''.join(seq)})
        rels = list({rel for seq in corpus.rels for rel in seq})
        tags = list({tag for seq in corpus.tags for tag in seq})
        vocab = cls(words, chars, tags, rels)

        return vocab

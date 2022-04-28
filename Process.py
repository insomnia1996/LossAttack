from typing import Sequence
from Vocab import Vocab
import pandas as pd
import torchtext
from torchtext import data
from Tokenize import tokenize
from Batch import MyIterator, batch_size_fn
import os
import pickle
import torch
from LossAttack.utils.corpus import Corpus
from transformers import BartTokenizer
from torch.utils.data import DataLoader, Dataset, Sampler


class TextSampler(Sampler):

    def __init__(self, lengths, batch_size, n_buckets, shuffle=False):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        # NOTE: the final bucket count is less than or equal to n_buckets
        self.sizes, self.buckets = kmeans(x=lengths, k=n_buckets)
        self.chunks = [max(size * len(bucket) // self.batch_size, 1)
                       for size, bucket in zip(self.sizes, self.buckets)]

    def __iter__(self):
        # if shuffle, shffule both the buckets and samples in each bucket
        range_fn = torch.randperm if self.shuffle else torch.arange
        for i in range_fn(len(self.buckets)):
            for batch in range_fn(len(self.buckets[i])).chunk(self.chunks[i]):
                yield [self.buckets[i][j] for j in batch.tolist()]

    def __len__(self):
        return sum(self.chunks)

class TextDataset(Dataset):

    def __init__(self, items, n_buckets=1):
        super(TextDataset, self).__init__()

        self.items = items

    def __getitem__(self, index):
        return tuple(item[index] for item in self.items)

    def __len__(self):#数据条数
        return len(self.items[0])

    @property
    def lengths(self):#每条数据分词长度
        return [len(i) for i in self.items[0]]
    
    @property
    def adv_length(self):#每条数据单词个数
        return [len(i) for i in self.items[1]]


    def check_correction(self):
        length = [len(i) for i in self.items[0]]#每条数据分词长度
        advlength = [len(i) for i in self.items[1]]#每条对抗数据分词长度
        for i in range(len(length)):
            assert length[i]==advlength[i], print("Data sample alignment wrong!")


def read_data(opt):
    
    if opt.src_data is not None:
        try:
            opt.src_data = open(opt.src_data,encoding='utf-8').read().strip().split('\n')
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()
    
    if opt.trg_data is not None:
        try:
            opt.trg_data = open(opt.trg_data,encoding='utf-8').read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()

def create_fields(opt):
    
    spacy_langs = ['en', 'fr', 'de', 'es', 'pt', 'it', 'nl']
    if opt.src_lang not in spacy_langs:
        print('invalid src language: ' + opt.src_lang + 'supported languages : ' + spacy_langs)  
    if opt.trg_lang not in spacy_langs:
        print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + spacy_langs)
    
    print("loading spacy tokenizers...")
    
    t_src = tokenize(opt.src_lang)
    t_trg = tokenize(opt.trg_lang)

    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()
        
    return(SRC, TRG)

def create_dataset_bak(opt, SRC, TRG):

    print("creating dataset and iterator... ")

    raw_data = {'src' : [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    
    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]

    df.to_csv("translate_transformer_temp.csv", index=False)
    
    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)
    device =torch.device('cuda' if torch.cuda.is_available() else "cpu")
    train_iter = MyIterator(train, batch_size=opt.batchsize, device=device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True, shuffle=True)
    
    os.remove('translate_transformer_temp.csv')

    if opt.load_weights is None:
        SRC.build_vocab(train)
        TRG.build_vocab(train)
        if opt.checkpoint > 0:
            try:
                os.mkdir("weights")
            except:
                print("weights folder already exists, run program with -load_weights weights to load them")
                quit()
            pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
            pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']

    opt.train_len = get_len(train_iter)

    return train_iter

def create_dataset(opt, number_path, sent_path, adv_path, train_or_dev='train'):

    print("creating %s dataset and iterator... " %train_or_dev)
    with open(number_path,'r') as f:
        line = f.readline().strip()#只有一行，存了攻击成功的index
        success_idx = line.split(" ")
    sent_corpus = Corpus.load_special(sent_path, success_idx)
    adv_corpus = Corpus.load_special(adv_path, success_idx)


    ###############data numericalization start########################
    sent_word, sent_arc, sent_rel=[],[],[]
    adv_word,adv_arc, adv_rel=[],[],[]
    barttkn=BartTokenizer.from_pretrained('facebook/bart-base', cache_dir='./data/pretrained/bart-base')
    opt.src_pad = barttkn.pad_token_id
    opt.trg_pad = barttkn.pad_token_id
    opt.train_len = len(sent_corpus.words)

    sent_vocab = Vocab.from_corpus(corpus=sent_corpus, min_freq=2)
    adv_vocab = Vocab.from_corpus(corpus=adv_corpus, min_freq=2)
    assert len(sent_corpus.words)==len(adv_corpus.words)
    assert len(sent_corpus.heads)==len(adv_corpus.heads)
    assert len(sent_corpus.rels)==len(adv_corpus.rels)

    for idx in range(len(sent_corpus.words)):
        seq = sent_corpus.words[idx]
        true_arc = sent_corpus.heads[idx]
        rel1 = sent_corpus.rels[idx]
        rel2 = adv_corpus.prels[idx][:-1]
        true_rel = [sent_vocab.rel_dict.get(rel, 0) for rel in rel1]
        if str(idx) in success_idx:
            adv = adv_corpus.words[idx][:-1]
            aarc = adv_corpus.pheads[idx][:-1]
            arel = [adv_vocab.rel_dict.get(rel, 0) for rel in rel2]
        else:
            adv = sent_corpus.words[idx]#原样本还原原样本
            aarc = sent_corpus.heads[idx]
            arel = [sent_vocab.rel_dict.get(rel, 0) for rel in rel1]
        
        print("original seq, arc & rel %d: " %idx, seq, true_arc, rel1)
        print("adversarial seq, arc & rel %d: " %idx, adv, aarc, rel2)
        
        #['<s>', 'The', 'idea', 'is', 'to', 'have', 'money', 'rolling', 'over', 'each', 'year', 'at', 'prevailing', 'interest', 'rates', '.', '</s>']
        words =barttkn.convert_tokens_to_ids(barttkn.tokenize(" ".join(seq).lower()))
        advs =barttkn.convert_tokens_to_ids(barttkn.tokenize(" ".join(adv).lower()))

        sent_word.append(torch.tensor(words))
        adv_word.append(torch.tensor(advs))

        sent_arc.append(torch.tensor(true_arc))
        adv_arc.append(torch.tensor(aarc))
        
        sent_rel.append(torch.tensor(true_rel))
        adv_rel.append(torch.tensor(arel))
            

    with open(os.path.join(opt.tensor_dir, train_or_dev, "ori_words.pkl"), 'wb') as f:
        pickle.dump(sent_word,f)
    with open(os.path.join(opt.tensor_dir, train_or_dev, "ori_arcs.pkl"), 'wb') as f:
        pickle.dump(sent_arc,f)
    with open(os.path.join(opt.tensor_dir, train_or_dev, "ori_rels.pkl"), 'wb') as f:
        pickle.dump(sent_rel,f)
    with open(os.path.join(opt.tensor_dir, train_or_dev, "adv_words.pkl"), 'wb') as f:
        pickle.dump(adv_word,f)
    with open(os.path.join(opt.tensor_dir, train_or_dev, "adv_arcs.pkl"), 'wb') as f:
        pickle.dump(adv_arc,f)
    with open(os.path.join(opt.tensor_dir, train_or_dev, "adv_rels.pkl"), 'wb') as f:
        pickle.dump(adv_rel,f)

    ###############data numericalization finish########################

    
    textset = TextDataset((sent_word, sent_arc, sent_rel, adv_word, adv_arc, adv_rel))
    device =torch.device('cuda' if torch.cuda.is_available() else "cpu")
    batch_sampler = TextSampler(lengths=textset.lengths,
                                batch_size=opt.batchsize,
                                n_buckets=1,
                                shuffle=True)#取index用sampler，不涉及数据
    loader = DataLoader(dataset=textset,
                        batch_sampler=batch_sampler,
                        collate_fn=collate_fn)

    return loader

def load_dataset(opt, number_path, sent_path, adv_path,  train_or_dev='train'):
    with open(number_path,'r') as f:
        line = f.readline().strip()#只有一行，存了攻击成功的index
        success_idx = line.split(" ")
    sent_corpus = Corpus.load_special(sent_path, success_idx)
    adv_corpus = Corpus.load_special(adv_path, success_idx)
    barttkn=BartTokenizer.from_pretrained('facebook/bart-base', cache_dir='./data/pretrained/bart-base')
    opt.src_pad = barttkn.pad_token_id
    opt.trg_pad = barttkn.pad_token_id
    opt.train_len = len(sent_corpus.words)
    print("loading %s dataset and iterator... " %train_or_dev)
    
    with open(os.path.join(opt.tensor_dir, train_or_dev, "ori_words.pkl"), 'rb') as f:
        sent_word = pickle.load(f)
    with open(os.path.join(opt.tensor_dir, train_or_dev, "ori_arcs.pkl"), 'rb') as f:
        sent_arc = pickle.load(f)
    with open(os.path.join(opt.tensor_dir, train_or_dev, "ori_rels.pkl"), 'rb') as f:
        sent_rel = pickle.load(f)
    with open(os.path.join(opt.tensor_dir, train_or_dev, "adv_words.pkl"), 'rb') as f:
        adv_word = pickle.load(f)
    with open(os.path.join(opt.tensor_dir, train_or_dev, "adv_arcs.pkl"), 'rb') as f:
        adv_arc = pickle.load(f)
    with open(os.path.join(opt.tensor_dir, train_or_dev, "adv_rels.pkl"), 'rb') as f:
        adv_rel = pickle.load(f)
    print("sample: ",barttkn.decode(sent_word[0]),barttkn.decode(sent_word[1]),barttkn.decode(sent_word[2]))
    #number = [torch.tensor([le], dtype=torch.int32) for le in range(opt.train_len)]
    textset = TextDataset((sent_word, sent_arc, sent_rel, adv_word, adv_arc, adv_rel))#number))
    device =torch.device('cuda' if torch.cuda.is_available() else "cpu")
    batch_sampler = TextSampler(lengths=textset.lengths,
                                batch_size=opt.batchsize,
                                n_buckets=1,
                                shuffle=True)#取index用sampler，不涉及数据
    loader = DataLoader(dataset=textset,
                        batch_sampler=batch_sampler,
                        collate_fn=collate_fn)

    return loader

def kmeans(x, k):
    x = torch.tensor(x, dtype=torch.float)
    # initialize k centroids randomly
    c, old = x[torch.randperm(len(x))[:k]], None
    # assign labels to each datapoint based on centroids
    dists, y = torch.abs_(x.unsqueeze(-1) - c).min(dim=-1)

    while old is None or not c.equal(old):
        # handle the empty clusters
        for i in range(k):
            # choose the farthest datapoint from the biggest cluster
            # and move that the empty cluster
            if not y.eq(i).any():
                mask = y.eq(torch.arange(k).unsqueeze(-1))
                lens = mask.sum(dim=-1)
                biggest = mask[lens.argmax()].nonzero().view(-1)
                farthest = dists[biggest].argmax()
                y[biggest[farthest]] = i
        # update the centroids
        c, old = torch.tensor([x[y.eq(i)].mean() for i in range(k)]), c
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(x.unsqueeze(-1) - c).min(dim=-1)
    clusters = [y.eq(i) for i in range(k)]
    clusters = [i.nonzero().view(-1).tolist() for i in clusters if i.any()]
    centroids = [round(x[i].mean().item()) for i in clusters]

    return centroids, clusters




def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    # type: (List[Tensor], bool, float) -> Tensor
    """
    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor
    return out_tensor


def collate_fn(data):
    reprs = (pad_sequence(i, True, padding_value=1) for i in zip(*data))

    if torch.cuda.is_available():
        reprs = (i.cuda() for i in reprs)

    return reprs

def get_len(train):

    for i, b in enumerate(train):
        pass
    
    return i

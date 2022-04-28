# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import copy


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
'''
def pad_sequence_bak(sequences, batch_first=False, padding_value=0.0):
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
    padding_cnt=[]
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        padding_cnt.append(max_len-length)#padding位数
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor
    return out_tensor, padding_cnt
'''

def pad_with_cnt(sequences, padding_cnt, batch_first=False):
    # type: (List[Tensor], list, bool) -> Tensor
    """
    Args:
        sequences (list[Tensor]): list of variable length sequences.
        padding_cnt(list[int]): list of padding count of each sentence in batch.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    assert len(padding_cnt)==len(sequences) #share same batch size
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) + padding_cnt[idx] for idx, s in enumerate(sequences)])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
    out_tensor = sequences[0].new_full(out_dims, 0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        one_dim = (padding_cnt[i],) + trailing_dims
        if batch_first:
            out_tensor[i, :length, ...] = tensor
            if padding_cnt[i]>0:
                out_tensor[i, length:length+padding_cnt[i], ...] = tensor.new_full(one_dim, 1)
                #print("Length after padding: ", out_tensor[i, ...])
        else:
            out_tensor[:length, i, ...] = tensor
            if padding_cnt[i]>0:
                out_tensor[length:length+padding_cnt[i], i, ...] = tensor.new_full(one_dim, 1)
                #print("Length after padding: ", out_tensor[:, i, ...])

    return out_tensor

def collate_fn(data):
    #batch padding，最后发现只需要pad 0即可
    reprs = (pad_sequence(i, True) for i in zip(*data))

    if torch.cuda.is_available():
        reprs = (i.cuda() for i in reprs)

    return reprs

def collate_fn_bak(data):
    #数据pad应该是1，synt vocab、barttkn的pad id 均为1
    #长度pad应该是0，不会影响长度计算
    #print(data)#12元元组的列表，列表长度为batchsize

    repr_lst=[]
    assert len(list(zip(*data)))==12
    for idx, i in enumerate(list(zip(*data))):#bsz元元组的列表，列表第0个元素是所有words序号的元组
        
        if idx!=5 or idx!=11:#长度pad为0
            tmp = pad_sequence(i, True, padding_value=1)
            repr_lst.append(tmp)
        else:#数据pad为1
            tmp = pad_sequence(i, True, padding_value=0)
            repr_lst.append(tmp)

    reprs = tuple(repr_lst)
    #reprs = (pad_sequence(i, True) for i in zip(*data))
    if torch.cuda.is_available():
        reprs = (i.cuda() for i in reprs)
    return reprs


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
    def sub_length(self):#每条数据单词个数
        return [len(i) for i in self.items[5]]

    def maxlen(self):
        return max([len(i) for i in self.items[0]] + [len(i) for i in self.items[6]])#原样本与对抗样本中最长bpe长度
    
    def max_syntlen(self):
        return max([len(i) for i in self.items[2]] + [len(i) for i in self.items[3]]
            + [len(i) for i in self.items[8]] + [len(i) for i in self.items[9]])

    def check_correction(self):
        length = [len(i) for i in self.items[0]]#每条数据分词长度
        sublen_sum = [torch.sum(i).item() for i in self.items[4]]
        for i in range(len(length)):
            assert length[i]==sublen_sum[i], print("Data sample alignment wrong!")
        
        tgt_length = [len(i) for i in self.items[5]]#每条对抗样本分词长度
        tgt_sublen_sum = [torch.sum(i).item() for i in self.items[9]]
        for i in range(len(tgt_length)):
            assert tgt_length[i]==tgt_sublen_sum[i], print("Adv sample alignment wrong!")
        

def batchify(dataset, batch_size, n_buckets=1, shuffle=False, collate_fn=collate_fn):
    batch_sampler = TextSampler(lengths=dataset.lengths,
                                batch_size=batch_size,
                                n_buckets=n_buckets,
                                shuffle=shuffle)#取index用sampler，不涉及数据
    loader = DataLoader(dataset=dataset,
                        batch_sampler=batch_sampler,
                        collate_fn=collate_fn)#_bak)

    return loader

# -*- coding: utf-8 -*-

from LossAttack.models.crfpar.modules import CHAR_LSTM, MLP, BertEmbedding, Biaffine, BiLSTM
from LossAttack.models.crfpar.modules.dropout import IndependentDropout, SharedDropout
from LossAttack.models.crfpar.utils.fn import pad

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from LossAttack.utils.vocab import Vocab
from LossAttack.utils.corpus import Corpus

class CRFParser(nn.Module):

    def __init__(self, args):
        super(CRFParser, self).__init__()
        self.args = args
        
        # the embedding layer
        ftrain = Corpus.load(args.ftrain)
        vocab = torch.load(args.vocab)
        args.update({
            'max_length':ftrain.maxlen(),
            'n_words': vocab.n_train_words,
            'n_tags': vocab.n_tags,
            'n_rels': vocab.n_rels,
            'n_chars': vocab.n_chars,
            'pad_index': vocab.pad_index,
            'unk_index': vocab.unk_index
        })
        
        self.word_embed = nn.Embedding(num_embeddings=args.n_words,
                                       embedding_dim=args.n_embed)
        
        self.feat_embed = nn.Embedding(num_embeddings=args.n_tags,
                                           embedding_dim=args.n_tag_embed)
        self.embed_dropout = IndependentDropout(p=args.embed_dropout)

        # the word-lstm layer
        self.lstm = BiLSTM(input_size=args.n_embed+args.n_tag_embed,
                           hidden_size=args.n_lstm_hidden,
                           num_layers=args.n_lstm_layers,
                           dropout=args.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)

        # the MLP layers
        self.mlp_arc_d = MLP(n_in=args.n_lstm_hidden*2,
                             n_out=args.n_mlp_arc,
                             dropout=args.mlp_dropout)
        self.mlp_arc_h = MLP(n_in=args.n_lstm_hidden*2,
                             n_out=args.n_mlp_arc,
                             dropout=args.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=args.n_lstm_hidden*2,
                             n_out=args.n_mlp_rel,
                             dropout=args.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=args.n_lstm_hidden*2,
                             n_out=args.n_mlp_rel,
                             dropout=args.mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=args.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=args.n_mlp_rel,
                                 n_out=args.n_rels,
                                 bias_x=True,
                                 bias_y=True)
        self.pad_index = args.pad_index
        self.unk_index = args.unk_index


    def forward(self, words, feats):
        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        # set the indices larger than num_embeddings to unk_index
        ext_mask = words.ge(self.word_embed.num_embeddings)
        ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        if self.args.feat == 'char':
            feat_embed = self.feat_embed(feats[mask])
            feat_embed = pad(feat_embed.split(lens.tolist()), total_length=seq_len)
        elif self.args.feat == 'bert':
            feat_embed = self.feat_embed(*feats)
        else:
            feat_embed = self.feat_embed(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), dim=-1)

        x = pack_padded_sequence(embed, lens.to("cpu"), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        # apply MLPs to the BiLSTM output states
        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    @classmethod
    def load(cls, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(state['args'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def save(self, path):
        state = {
            'config': self.args,
            'state_dict': self.state_dict(),
            'embeddings': self.word_embed.weight
        } 
        torch.save(state, path)

# -*- coding: utf-8 -*-
from LossAttack.libs.luna import fetch_best_ckpt_name
from LossAttack.models.modules import (MLP, Biaffine, BiLSTM, EmbeddingDropout,
                                     SharedDropout)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class WordParser(nn.Module):

    def __init__(self, config, embeddings):
        super(WordParser, self).__init__()

        self.config = config
        # the embedding layer
        #self.embed = nn.Embedding(num_embeddings=config.n_train_words,
        #                              embedding_dim=config.n_embed)
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=False)

        self.embed_dropout = EmbeddingDropout(p=config.embed_dropout)

        # the word-lstm layer
        self.lstm = BiLSTM(input_size=config.n_embed,
                           hidden_size=config.n_lstm_hidden,
                           num_layers=config.n_lstm_layers,
                           dropout=config.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=config.lstm_dropout)

        # the MLP layers
        self.mlp_arc_h = MLP(n_in=config.n_lstm_hidden*2,
                             n_hidden=config.n_mlp_arc,
                             dropout=config.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=config.n_lstm_hidden*2,
                             n_hidden=config.n_mlp_arc,
                             dropout=config.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=config.n_lstm_hidden*2,
                             n_hidden=config.n_mlp_rel,
                             dropout=config.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=config.n_lstm_hidden*2,
                             n_hidden=config.n_mlp_rel,
                             dropout=config.mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=config.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=config.n_mlp_rel,
                                 n_out=config.n_rels,
                                 bias_x=True,
                                 bias_y=True)
        self.pad_index = config.pad_index
        self.unk_index = config.unk_index

        self.reset_parameters()

    def reset_parameters(self):
        pass
        # nn.init.zeros_(self.embed.weight)

    def forward(self, words, tags):
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        # set the indices larger than num_embeddings to unk_index
        # ext_mask = words.ge(self.pretrained.num_embeddings)
        # ext_mask = words.ge(self.embed.num_embeddings)
        # ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        # embed = self.pretrained(words) + self.embed(ext_words)
        embed = self.embed(words)
        # tag_embed = self.tag_embed(tags)
        #embedtag_embed = self.embed_dropout(embed, tag_embed)
        # concatenate the word and tag representations
        #x = torch.cat((embed, tag_embed), dim=-1)
        x = self.embed_dropout(embed)

        sorted_lens, indices = torch.sort(lens, descending=True)
        inverse_indices = indices.argsort()
        x = pack_padded_sequence(x[indices], sorted_lens.cpu(), True)
        x = self.lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.lstm_dropout(x)[inverse_indices]

        # apply MLPs to the BiLSTM output states
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    @classmethod
    def load(cls, fname):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        state = torch.load(fname, map_location=device)
        parser = cls(state['config'], state['embeddings'])
        parser.load_state_dict(state['state_dict'])
        parser.to(device)

        return parser

    def save(self, fname):
        state = {
            'config': self.config,
            'embeddings': self.embed.weight,
            'state_dict': self.state_dict()
        }
        torch.save(state, fname)

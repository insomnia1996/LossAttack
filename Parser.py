import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class SharedDropout(nn.Module):

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first

    def extra_repr(self):
        info = f"p={self.p}"
        if self.batch_first:
            info += f", batch_first={self.batch_first}"

        return info

    def forward(self, x):
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batch_first else mask

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_full(x.shape, 1 - p)
        mask = torch.bernoulli(mask) / (1 - p)

        return mask


class IndependentDropout_DEC(nn.Module):

    def __init__(self, p=0.5):
        super(IndependentDropout_DEC, self).__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, x, y, eps=1e-12):
        if self.training:
            x_mask = torch.bernoulli(x.new_full(x.shape[:2], 1 - self.p))
            y_mask = torch.bernoulli(y.new_full(y.shape[:2], 1 - self.p))
            scale = 2.0 /(torch.cat((x_mask , y_mask), dim=1) + eps)
            x_mask *= scale
            y_mask *= scale

            x *= x_mask.unsqueeze(dim=-1)
            y *= y_mask.unsqueeze(dim=-1)


        return x, y

class IndependentDropout(nn.Module):

    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, x, y, eps=1e-12):
        if self.training:
            x_mask = torch.bernoulli(x.new_full(x.shape[:2], 1 - self.p))
            y_mask = torch.bernoulli(y.new_full(y.shape[:2], 1 - self.p))

            scale = 2.0 / (x_mask + y_mask + eps)
            x_mask *= scale
            y_mask *= scale

            x *= x_mask.unsqueeze(dim=-1)
            y *= y_mask.unsqueeze(dim=-1)

        return x, y


class EmbeddingDropout(nn.Module):

    def __init__(self, p=0.5):
        super(EmbeddingDropout, self).__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"


    def forward(self, x, eps=1e-12):
        if self.training:
            x_mask = torch.bernoulli(x.new_full(x.shape[:2], 1 - self.p))
            x *= x_mask.unsqueeze(dim=-1)
        return x

class MLP(nn.Module):

    def __init__(self, n_in, n_hidden, dropout=0):
        super(MLP, self).__init__()

        self.linear = nn.Linear(n_in, n_hidden)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        for layer in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            self.b_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            input_size = hidden_size * 2

        self.reset_parameters()

    def reset_parameters(self):
        for i in self.parameters():
            # apply orthogonal_ to weight
            if len(i.shape) > 1:
                nn.init.orthogonal_(i)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(i)

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        h, c = hx
        init_h, init_c = h, c
        output, seq_len = [], len(x)
        steps = reversed(range(seq_len)) if reverse else range(seq_len)
        if self.training:
            hid_mask = SharedDropout.get_mask(h, self.dropout)

        for t in steps:
            last_batch_size, batch_size = len(h), batch_sizes[t]
            if last_batch_size < batch_size:
                h = torch.cat((h, init_h[last_batch_size:batch_size]))
                c = torch.cat((c, init_c[last_batch_size:batch_size]))
            else:
                h = h[:batch_size]
                c = c[:batch_size]
            h, c = cell(input=x[t], hx=(h, c))
            output.append(h)
            if self.training:
                h = h * hid_mask[:batch_size]
        if reverse:
            output.reverse()
        output = torch.cat(output)

        return output

    def forward(self, x, hx=None):
        x, batch_sizes = x[0], x[1]
        batch_size = batch_sizes[0]

        if hx is None:
            init = x.new_zeros(batch_size, self.hidden_size)
            hx = (init, init)

        for layer in range(self.num_layers):
            if self.training:
                mask = SharedDropout.get_mask(x[:batch_size], self.dropout)
                mask = torch.cat([mask[:batch_size]
                                  for batch_size in batch_sizes])
                x *= mask
            x = torch.split(x, batch_sizes.tolist())
            f_output = self.layer_forward(x=x,
                                          hx=hx,
                                          cell=self.f_cells[layer],
                                          batch_sizes=batch_sizes,
                                          reverse=False)
            b_output = self.layer_forward(x=x,
                                          hx=hx,
                                          cell=self.b_cells[layer],
                                          batch_sizes=batch_sizes,
                                          reverse=True)
            x = torch.cat([f_output, b_output], -1)
        x = PackedSequence(x, batch_sizes)

        return x



class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        info = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            info += f", bias_x={self.bias_x}"
        if self.bias_y:
            info += f", bias_y={self.bias_y}"

        return info

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat([x, x.new_ones(x.shape[:-1]).unsqueeze(-1)], -1)
        if self.bias_y:
            y = torch.cat([y, y.new_ones(y.shape[:-1]).unsqueeze(-1)], -1)
        # [batch_size, 1, seq_len, d]
        x = x.unsqueeze(1)
        # [batch_size, 1, seq_len, d]
        y = y.unsqueeze(1)
        # [batch_size, n_out, seq_len, seq_len]
        s = x @ self.weight @ y.transpose(-1, -2)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s

class WordParser(nn.Module):

    def __init__(self, config, n_rels):
        super(WordParser, self).__init__()

        self.config = config
        # the embedding layer
        self.embed = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.d_model)

        self.embed_dropout = EmbeddingDropout(p=0.3)

        # the word-lstm layer
        self.lstm = BiLSTM(input_size=config.d_model,
                           hidden_size=config.d_model,
                           num_layers=config.encoder_layers,
                           dropout=0.3)
        self.lstm_dropout = SharedDropout(p=0.3)

        # the MLP layers
        self.mlp_arc_h = MLP(n_in=config.d_model*2,
                             n_hidden=config.d_model,
                             dropout=0.3)
        self.mlp_arc_d = MLP(n_in=config.d_model*2,
                             n_hidden=config.d_model,
                             dropout=0.3)
        self.mlp_rel_h = MLP(n_in=config.d_model*2,
                             n_hidden=config.d_model,
                             dropout=0.3)
        self.mlp_rel_d = MLP(n_in=config.d_model*2,
                             n_hidden=config.d_model,
                             dropout=0.3)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=config.d_model,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=config.d_model,
                                 n_out=n_rels,
                                 bias_x=True,
                                 bias_y=True)
        self.pad_index = config.pad_token_id

        self.reset_parameters()

    def reset_parameters(self):
        pass
        # nn.init.zeros_(self.embed.weight)

    def forward(self, words):
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

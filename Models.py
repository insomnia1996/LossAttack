from Parser import WordParser
import torch
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable 
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm
import copy
from Batch import create_masks
from Beam import k_best_outputs, init_beam
from transformers.models.bart.modeling_bart import (
    BartEncoder, 
    BartDecoder,
    BartConfig,
    BartLearnedPositionalEmbedding
)
from syntemb import SyntEmb

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer_bak(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

class Transformer(nn.Module):
    def __init__(self, config, trg_vocab):
        super().__init__()
        self.encoder = BartEncoder(config)#.from_pretrained('facebook/bart-base', cache_dir="/home/lyt/LossAttack/data/pretrained/bart-base")
        config.vocab_size = trg_vocab
        self.decoder = BartDecoder(config)#.from_pretrained('facebook/bart-base', cache_dir="/home/lyt/LossAttack/data/pretrained/bart-base")
        self.out = nn.Linear(config.d_model, config.vocab_size)
        #print("d_model, vocab_size:",config.d_model, config.vocab_size)
        n_rels=47
        max_length=155
        n_tags=49
        self.syntemb = SyntEmb(config, max_length)
        self.syntemb2 = SyntEmb(config, n_rels)
        #self.parser = WordParser(config, n_rels)

        config.is_encoder_decoder = True
        self.pad_token_id = config.pad_token_id
        print(config)
    
    def forward(self, src, src_arc, src_rel, trg, src_mask, trg_mask):
        e_outputs = self.encoder(input_ids=src, attention_mask=src_mask.squeeze(1))
        #src_arc, src_rel = self.parser(src)
        
        #src_arc = src_arc.argmax(-1)
        #src_rel = src_rel.argmax(-1)
        #src_rel = src_rel.gather(-1, src_arc.unsqueeze(-1)).squeeze(-1)
        arc_hidden = self.syntemb(src_arc)
        rel_hidden = self.syntemb2(src_rel)
        synt_hidden = torch.cat((arc_hidden, rel_hidden), dim=1)
        enc_hidden = torch.cat((e_outputs[0], synt_hidden), dim=1)
        #enc_hidden = e_outputs[0]
        #print("encoder output: ", enc_hidden.shape)
        d_outputs = self.decoder(input_ids=trg, attention_mask=trg_mask.squeeze(1), encoder_hidden_states=enc_hidden)
        d_outputs = d_outputs[0]
        #print("decoder output: ", d_outputs.shape)
        output = self.out(d_outputs)#(bsz, seq_len, vocab_size)
        return output
    
    def encode(self, src, src_arc, src_rel, src_mask):
        e_outputs = self.encoder(input_ids=src, attention_mask=src_mask.squeeze(1))
        #src_arc, src_rel = self.parser(src)
        #mask = self.get_mask(src, self.pad_token_id,punct_list=[2, 3, 5, 6, 7, 8, 25, 26, 27, 28, 29, 34, 35, 1954, 1955, 1959])
        #src_arc = src_arc[:,mask.squeeze(0)].argmax(dim=2).contiguous()
        #src_rel = src_rel[:,mask.squeeze(0)].argmax(dim=2).contiguous()
        arc_hidden = self.syntemb(src_arc)
        rel_hidden = self.syntemb2(src_rel)
        synt_hidden = torch.cat((arc_hidden, rel_hidden), dim=1)
        enc_hidden = torch.cat((e_outputs[0], synt_hidden), dim=1)
        
        return enc_hidden
    
    def decode(self, enc_hidden, trg, trg_mask):
        d_outputs = self.decoder(input_ids=trg, attention_mask=trg_mask.squeeze(1), encoder_hidden_states=enc_hidden)
        d_outputs = d_outputs[0]
        #print("decoder output: ", d_outputs.shape)
        output = self.out(d_outputs)#(bsz, seq_len, vocab_size)
        return output

    def predict(self, src, src_arc, src_rel, opt):#均为[1,seq_len]的tensor
        #src_arc, src_rel = self.parser(src)
        #mask = self.get_mask(src, self.pad_token_id,punct_list=[2, 3, 5, 6, 7, 8, 25, 26, 27, 28, 29, 34, 35, 1954, 1955, 1959])
        #src_arc = src_arc[:,mask.squeeze(0)].argmax(dim=2).contiguous()
        #src_rel = src_rel[:,mask.squeeze(0)].argmax(dim=2).contiguous()
        trg_input, enc_hidden, log_scores = init_beam(src, src_arc, src_rel, self, opt)
        
        eos_token = 2
        bos_token = 0
        ind = None
        for i in range(2, opt.max_strlen):
            src_mask, trg_mask = create_masks(src, trg_input[:,:i], opt)
            out = self.decode(enc_hidden, trg_input[:,:i], trg_mask)

            out = F.softmax(out, dim=-1)
            #print(out.shape)
            #trg_input = torch.argmax(out, dim=-1)
            trg_input, log_scores = k_best_outputs(trg_input, out, log_scores, i, opt.k)
            #print(trg_input.shape)
            
            ones = (trg_input==eos_token).nonzero() # Occurrences of end symbols for all input sentences.
            sentence_lengths = torch.zeros(len(trg_input), dtype=torch.long).cuda()
            for vec in ones:
                i = vec[0]
                if sentence_lengths[i]==0: # First end symbol has not been found yet
                    sentence_lengths[i] = vec[1] # Position of first end symbol

            num_finished_sentences = len([s for s in sentence_lengths if s > 0])

            if num_finished_sentences == opt.k:
                alpha = 0.7
                div = 1/(sentence_lengths.type_as(log_scores)**alpha)
                _, ind = torch.max(log_scores * div, 1)
                ind = ind.data[0]
                break
        if ind is None:
            if eos_token in trg_input[0]:
                length = (trg_input[0]==eos_token).nonzero()[0]
                return trg_input[0][1:length]
            else:
                return trg_input[0][1:]
        
        else:
            if eos_token in trg_input[ind]:
                length = (trg_input[ind]==eos_token).nonzero()[0]
                return trg_input[ind][1:length]
            else:
                return trg_input[ind][1:]

    def nopeak_mask(self, size, opt):
        np_mask = np.triu(np.ones((1, size, size)),
            k=1).astype('uint8')
        np_mask =  Variable(torch.from_numpy(np_mask) == 0)
        if opt.device >= 0:
            np_mask = np_mask.cuda()
        return np_mask

    def get_mask(self, words, pad_index, punct = False, punct_list = None):
        '''
        get the mask of a sentence, mask all <pad>
        the starting of sentence is <ROOT>, mask[0] is always False
        :param words: sentence
        :param pad_index: pad index
        :param punct: whether to ignore the punctuation, when punct is False, take all the punctuation index to False(for evaluation)
        punct is True for getting loss
        :param punct_list: only used when punct is False
        :return:
        For example, for a sentence:  <ROOT>     no      ,       it      was     n't     Black   Monday  .
        when punct is True,
        The returning value is       [False    True     True    True    True    True    True    True    True]
        when punct is False,
        The returning value is      [False    True     False    True    True    True    True    True    False]
        '''
        mask = words.ne(pad_index)
        mask[:, 0] = False
        if not punct:
            puncts = words.new_tensor(punct_list)
            mask &= words.unsqueeze(-1).ne(puncts).all(-1)
        return mask

def get_model(opt, src_vocab, trg_vocab):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1
    #MODIFIED
    config = BartConfig(vocab_size=src_vocab, d_model=opt.d_model, encoder_layers=opt.n_layers, decoder_layers=opt.n_layers,
                encoder_attention_heads=opt.heads, decoder_attention_heads=opt.heads, dropout=opt.dropout)
    model = Transformer(config, trg_vocab)
    #model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    
       
    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    if opt.device >= 0:
        model = model.cuda()
    
    return model
    
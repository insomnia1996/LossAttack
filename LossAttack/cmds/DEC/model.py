# -*- coding: utf-8 -*-
from torch.nn.modules.sparse import Embedding
from LossAttack.libs.luna import fetch_best_ckpt_name
from LossAttack.models.modules import (MLP, IndependentDropout_DEC, SharedDropout)
from LossAttack.models import WordParser, word
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PretrainedBartModel, 
    BartModel,
    BartConfig
)
import numpy as np
from transformers.models.bart.modeling_bart import (
    BartEncoder,
    BartDecoder,
    BartForConditionalGeneration,
)
from LossAttack.cmds.DEC.syntemb import SyntEmb

class DECParser(PretrainedBartModel):

    def __init__(self, config):
        
        config.update({'n_train_words': 50265})
        self.cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #init PLM config
        bartconfig = self.init_config(config)
        super().__init__(bartconfig)
        # the embedding layer
        self.embed = nn.Embedding(num_embeddings=config.n_train_words,
                                      embedding_dim=config.n_embed)
        self.tag_embed = nn.Embedding(num_embeddings=config.n_tags,
                                      embedding_dim=config.n_tag_embed)
        self.rel_embed = nn.Embedding(num_embeddings=config.n_rels,
                                      embedding_dim=config.n_tag_embed)
        self.arc_embed = nn.Embedding(num_embeddings=config.max_length,
                                      embedding_dim=config.n_tag_embed)    
        self.embed_dropout = IndependentDropout_DEC(p=config.embed_dropout)
        print("==== loading model ====")

        self.encoder = BartEncoder(bartconfig)
        self.decoder = BartDecoder(bartconfig)
        #self.embed = self.seq2seq.get_input_embeddings()

        # the MLP layers
        #self.mlp_word = MLP(n_in=bartconfig.d_model,
        #                     n_hidden=bartconfig.vocab_size,
        #                     dropout=config.mlp_dropout)
        self.mlp_word = nn.Linear(bartconfig.d_model, bartconfig.vocab_size)
        self.mlp_arc = MLP(n_in=bartconfig.d_model,
                             n_hidden=config.max_length,
                             dropout=config.mlp_dropout)
        self.mlp_rel = MLP(n_in=bartconfig.d_model,
                             n_hidden=config.n_rels,
                             dropout=config.mlp_dropout)
        self.syntemb = SyntEmb(bartconfig, config.max_length)
        self.syntemb2 = SyntEmb(bartconfig, config.n_rels)
    

    def init_config(self, args):
        print("==== loading config ====")
        config = BartConfig.from_pretrained('facebook/bart-base', cache_dir=args.cache_dir)
        config.d_model = args.n_embed
        #config.encoder_layers =8
        #config.decoder_layers =8
        #config.encoder_attention_heads =8
        #config.decoder_attention_heads =8
        config.vocab_size = 50265#args.n_train_words
        config.max_position_embeddings = max(args.max_length, args.max_synt_len)


        config.is_encoder_decoder = True
        self.pad_token_id = config.pad_token_id
        print(config)
        return config

    def forward(self, words, tags, arcs, rels, tree, tgt_words, tgt_tags, tgt_arcs, tgt_rels, tgt_tree):#adv sample false input & ground truth input
        # get the mask and lengths of given batch
        
        word_embed = self.embed(words)
        src_mask = words != self.pad_token_id
        x = word_embed#encoder input

        tgt_in = tgt_words[:, :-1].contiguous()
        tgt_mask = tgt_in != self.pad_token_id
        tgt_word_embed = self.embed(tgt_in)
        y = tgt_word_embed#[bsz, seq_len, embed_dim] decoder input
        

        labels = tgt_words[:, 1:].contiguous()
        labels_arcs = tgt_arcs[:, 1:].contiguous()
        labels_rels = tgt_rels[:, 1:].contiguous()

        encoder_outputs = self.encoder(
                inputs_embeds=x,
                attention_mask=src_mask
            )
        arc_hidden = self.syntemb(arcs)
        rel_hidden = self.syntemb2(rels)
        synt_hidden = torch.cat((arc_hidden, rel_hidden), dim=1)
        enc_hidden = encoder_outputs[0]#torch.cat((encoder_outputs[0], synt_hidden), dim=1)
        outputs = self.decoder(
            inputs_embeds=y,
            attention_mask=tgt_mask,
            encoder_hidden_states=enc_hidden,
            return_dict=True)

        hidden = outputs.last_hidden_state
        logits_out = self.mlp_word(hidden)
        logits_rels = self.mlp_rel(hidden)
        logits_arcs = self.mlp_arc(hidden)
        
        return logits_out, logits_arcs, logits_rels, labels, labels_arcs, labels_rels

    def nopeak_mask(self, size, is_cuda):
        np_mask = np.triu(np.ones((1, size, size)),
            k=1).astype('uint8')
        np_mask =  torch.Variable(torch.from_numpy(np_mask) == 0)
        if is_cuda:
            np_mask = np_mask.cuda()
        return np_mask

    def create_masks(self, src, trg, opt):
        
        src_mask = (src != self.pad_token_id).unsqueeze(-2)
        #(bsz, seq_len)
        seq_len = src_mask.size(-1)
        src_mask = src_mask.unsqueeze(-1).repeat(1,1,seq_len).contiguous()
        #(bsz, seq_len, seq_len)
        src_mask = src_mask.unsqueeze(1)
        if trg is not None:
            trg_mask = (trg != self.pad_token_id).unsqueeze(-2)
            size = trg.size(1) # get seq_len for matrix
            np_mask = self.nopeak_mask(size, trg.is_cuda())
            if trg.is_cuda:
                np_mask.cuda()
            trg_mask = trg_mask & np_mask
            trg_mask = trg_mask.unsqueeze(1)
            
        else:
            trg_mask = None
        print("mask shape: ", src_mask.shape, trg_mask.shape)
        return src_mask, trg_mask

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


#temporarily unused
'''
class DECParser_bak(BartPretrainedModel):

    def __init__(self, config):
        
        #init config index
        self.pad_index = config.pad_index
        self.unk_index = config.unk_index
        self.root_index = 1956
        self.dot_index = 34

        self.cuda_device = self.cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #init PLM config
        bartconfig = self.init_config(config)
        super().__init__(bartconfig)
        # the embedding layer
        self.embed = nn.Embedding(num_embeddings=config.n_train_words,
                                      embedding_dim=config.n_embed)
        self.tag_embed = nn.Embedding(num_embeddings=config.n_tags,
                                      embedding_dim=config.n_tag_embed)
        self.rel_embed = nn.Embedding(num_embeddings=config.n_rels,
                                      embedding_dim=config.n_tag_embed)
        self.arc_embed = nn.Embedding(num_embeddings=config.max_length,
                                      embedding_dim=config.n_tag_embed)              
        self.embed_dropout = IndependentDropout_DEC(p=config.embed_dropout)
        print("==== loading model ====")
        self.seq2seq = BartModel(bartconfig).to(self.cuda_device)#.from_pretrained('facebook/bart-base', cache_dir=config.cache_dir).to(self.cuda_device)
        #self.seq2seq.zero_grad()


        # the MLP layers
        self.mlp_word = MLP(n_in=bartconfig.d_model,
                             n_hidden=bartconfig.vocab_size,
                             dropout=config.mlp_dropout)

        self.word_parser = WordParser(config)
        self.mlp_arc = MLP(n_in=bartconfig.d_model,
                             n_hidden=config.max_length,
                             dropout=config.mlp_dropout)
        self.mlp_rel = MLP(n_in=bartconfig.d_model,
                             n_hidden=config.n_rels,
                             dropout=config.mlp_dropout)


    def calculate_loss_and_accuracy(self, logits, labels, device):
        """
        计算非pad_id的平均loss和准确率
        :param logits:
        :param labels:
        :param device:
        :return:
        """
        # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,vocab_size]
        # 用前n-1个token，预测出第n个token
        # 用第i个token的prediction_score用来预测第i+1个token。
        # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，shift_labels表示第[1，n-1]的label

        shift_logits = logits[..., :-1, :].contiguous().to(device)
        shift_labels = labels[..., 1:].contiguous().to(device)
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_index, reduction='sum')  # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))

        _, preds = shift_logits.max(dim=-1)  # preds表示对应的prediction_score预测出的token在vocab中的id。维度为[batch_size,token_len]

        # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
        not_ignore = shift_labels.ne(self.pad_index)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
        num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

        correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的token
        correct = correct.float().sum()

        accuracy = correct / num_targets
        loss = loss / num_targets
        return loss, accuracy

    def init_config(self, args):
        print("==== loading config ====")
        config = BartConfig.from_pretrained('facebook/bart-base', cache_dir=args.cache_dir)
        config.vocab_size = args.n_train_words
        config.max_position_embeddings = args.max_length
        config.d_model = args.n_embed#3*args.n_tag_embed+args.n_embed
        config.dropout = args.mlp_dropout
        config.bos_token_id = self.root_index
        config.eos_token_id = 0 #self.dot_index
        config.encoder_attention_heads = 8
        config.decoder_attention_heads = 8
        config.forced_eos_token_id = self.dot_index
        config.decoder_start_token_id = self.root_index
        config.is_encoder_decoder = True
        print(config)
        return config

    def forward(self, words, tags, arcs, rels, tgt_words, tgt_tags, tgt_arcs, tgt_rels):#adv sample false input & ground truth input
        # get the mask and lengths of given batch
        #print("model in shape: ",words.shape, arcs.shape, tgt_words.shape, tgt_rels.shape)
        mask = words.ne(self.pad_index)#mask掉<pad>位置输入
        seq_len = mask.sum(dim=1)
        bsz = words.shape[0]
        # get outputs from embedding layers
        # embed = self.pretrained(words) + self.embed(ext_words)
        word_embed = self.embed(words)
        tag_embed = self.tag_embed(tags)
        arc_embed = self.arc_embed(arcs)
        rel_embed = self.rel_embed(rels) 
        word_embed, tag_embed, arc_embed, rel_embed = self.embed_dropout(word_embed, tag_embed, arc_embed, rel_embed)
        # concatenate the word and tag representations

        x=word_embed
        #x = torch.cat((word_embed, tag_embed, arc_embed, rel_embed), dim=-1)#[bsz, seq_len, embed_dim]
        x = word_embed + tag_embed + arc_embed + rel_embed
        tgt_word_embed = self.embed(tgt_words)
        tgt_tag_embed = self.tag_embed(tgt_tags)
        tgt_arc_embed = self.arc_embed(tgt_arcs)
        tgt_rel_embed = self.rel_embed(tgt_rels) 
        tgt_word_embed, tgt_tag_embed, tgt_arc_embed, tgt_rel_embed = self.embed_dropout(tgt_word_embed, tgt_tag_embed, tgt_arc_embed, tgt_rel_embed)

        #y = torch.cat((tgt_word_embed, tgt_tag_embed, tgt_arc_embed, tgt_rel_embed), dim=-1)#[bsz, dec_seq_len, embed_dim]
        y = tgt_word_embed + tgt_tag_embed + tgt_arc_embed + tgt_rel_embed
        pad = torch.empty((bsz, 1), dtype = torch.int64, device =words.device).fill_(0)
        labels = torch.cat((tgt_words[:, 1:],pad),dim=-1).contiguous()
        labels_arcs = torch.cat((tgt_arcs[:, 1:],pad),dim=-1).contiguous()
        labels_rels = torch.cat((tgt_rels[:, 1:],pad),dim=-1).contiguous()

        outputs = self.seq2seq(inputs_embeds=x, decoder_inputs_embeds=y)
        decoder_out = outputs.last_hidden_state #(batch_size, sequence_length, hidden_size)
        # apply MLPs to BART hidden states
        logits_out = self.mlp_word(decoder_out)
        
        #应该用其他方式计算
        s_arc, s_rel = self.word_parser(torch.argmax(logits_out,dim=-1))
        heads = s_arc.argmax(dim=2)
        logits_arcs = s_arc
        logits_deprel = s_rel
        gather_index = heads.view(*heads.size(), 1, 1).expand(-1, -1, -1, logits_deprel.size(-1))
        logits_rels = torch.gather(logits_deprel, dim=2, index=gather_index).contiguous().squeeze(2)
        #print(logits_arcs.shape,logits_rels.shape)
        #[bsz, seq_len, n_arcs(seq_len)], [bsz, seq_len, n_rels]
        #logits_arcs = self.mlp_arc(decoder_out)
        #logits_rels = self.mlp_rel(decoder_out)
        
        loss1, acc1 = self.calculate_loss_and_accuracy(logits_out, labels, device=self.cuda_device)
        loss2, acc2 = self.calculate_loss_and_accuracy(logits_arcs, labels_arcs, device=self.cuda_device) 
        loss3, acc3 = self.calculate_loss_and_accuracy(logits_rels, labels_rels, device=self.cuda_device)

        loss = loss1+loss2+loss3
        acc = (acc1+acc2+acc3)/3.0

        return loss, acc, logits_out, logits_arcs, logits_rels


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
'''




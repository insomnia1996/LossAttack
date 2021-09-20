import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from transformers import PretrainedBartModel  
from transformers.models.bart.modeling_bart import (
    BartEncoderLayer, 
    BartLearnedPositionalEmbedding
)

class SyntEmb(PretrainedBartModel):
    def __init__(self, config, maxlen):
        super().__init__(config)

        self.encoder = ParaBartEncoder(config, maxlen)
        
        self.init_weights()

    def forward(
        self,
        input_ids,      
        attention_mask=None,
        decoder_padding_mask=None,
        encoder_outputs=None,
        return_encoder_outputs=False,
    ):
        if attention_mask is None:
            attention_mask = input_ids == self.config.pad_token_id #1
            #(bsz, seq_len)
            seq_len = attention_mask.size(-1)
            attention_mask = attention_mask.unsqueeze(-1).repeat(1,1,seq_len).contiguous()
            #(bsz, seq_len, seq_len)
            attention_mask = attention_mask.unsqueeze(1)
            
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        return encoder_outputs
    
    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        encoder_outputs = past[0]
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_input_ids": torch.cat((decoder_input_ids, torch.zeros((decoder_input_ids.shape[0], 1), dtype=torch.long).cuda()), 1),
            "attention_mask": attention_mask,
        }

    def get_encoder(self):
        return self.encoder
    
    @staticmethod
    def _reorder_cache(past, beam_idx):
        enc_out = past[0][0]

        new_enc_out = enc_out.index_select(0, beam_idx)

        past = ((new_enc_out, ), )
        return past


class ParaBartEncoder(nn.Module):
    def __init__(self, config, maxlen):
        super().__init__()
        self.config = config

        self.dropout = config.dropout
        self.embed_synt = nn.Embedding(maxlen, config.d_model, config.pad_token_id)
        self.embed_synt.weight.data.normal_(mean=0.0, std=config.init_std)
        self.embed_synt.weight.data[config.pad_token_id].zero_()

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model
        )
        
        self.synt_layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(1)])

        self.synt_layernorm_embedding = LayerNorm(config.d_model)
        
        self.pooling = MeanPooling(config)
        
    def forward(self, input_synt_ids, attention_mask):
        encoder_outputs = self.forward_synt(input_synt_ids, attention_mask)
        return encoder_outputs
        
    def forward_synt(self, input_synt_ids, attention_mask):
        input_synt_embeds = self.embed_synt(input_synt_ids) + self.embed_positions(input_synt_ids.shape)        
        y = self.synt_layernorm_embedding(input_synt_embeds)        
        y = F.dropout(y, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        #y = y.transpose(0, 1)
        for encoder_synt_layer in self.synt_layers:
            y, _ = encoder_synt_layer(y, attention_mask=attention_mask, layer_head_mask=None, output_attentions=True)
        # T x B x C -> B x T x C
        #y = y.transpose(0, 1)
        return y
        

    def embed(self, input_token_ids, attention_mask=None, pool='mean'):
        if attention_mask is None:
            attention_mask = input_token_ids == self.config.pad_token_id
            
        x = self.forward_token(input_token_ids, attention_mask)
        
        sent_embeds = self.pooling(x, input_token_ids)
        return sent_embeds
            
class MeanPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, x, input_token_ids):
        mask = input_token_ids != self.config.pad_token_id
        mean_mask = mask.float()/mask.float().sum(1, keepdim=True)
        x = (x*mean_mask.unsqueeze(2)).sum(1, keepdim=True)
        return x
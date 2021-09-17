from transformers import BertTokenizer, BertForMaskedLM, BartTokenizer, BartModel, BartConfig,BartForConditionalGeneration
import torch

tokenizer2 = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir = "./data/pretrained/bart-base")
device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

article2 = "rolls-royce motor cars inc. said it expects its u.s. sales to get steady at about 1,200 cars in 1990."
dec_str = tokenizer2.tokenize("<s>"+article2+"</s>")
print(tokenizer2.eos_token_id,tokenizer2.pad_token_id,tokenizer2.bos_token_id, tokenizer2.mask_token_id)
#2 1 0 50264
print(dec_str)
context=[tokenizer2.convert_tokens_to_ids(str1) for str1 in dec_str]
print(tokenizer2.encode(article2), tokenizer2(article2))
print(context)
print(tokenizer2.decode([    0,    51,   429,   190,  1149,    62,  1414,    11,    10,  3064,
          210,  2156,    37,    26,  2156,    11,    41,  1351,     7,  3014,
            5,  3041,     9,  1093,  2081,    88,    49, 28696,  4154,   530,
        15698,   479,  1437,     2,     1,     1,     1,     1,     1,     1,
            1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
            1,     1,     1,     1]))
print(tokenizer2.decode([    0,   252,   429,   190,  1149,    62,   647,    11,    10,  3064,
          210,  2156,    37,    26,  2156,    11,    41,  1351,     7,  3014,
            5,  3041,     9,  1093,  2081,    88,    49,  6110,   281, 12755,
          479,  1437,     2,     1,     1,     1,     1,     1,     1,     1,
            1,     1,     1,     1,     1,     1,     1,     1]))

#TODO: 把corenlp加到lossattack vocab.numericalize里面，加入语法树（需要deleaf）
from stanfordcorenlp import StanfordCoreNLP

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

def is_paren(tok):
    return tok == ")" or tok == "("
def deleaf(tree):
    nonleaves = ''
    for w in tree.replace('\n', '').split():
        w = w.replace('(', '( ').replace(')', ' )')
        nonleaves += w + ' '

    arr = nonleaves.split()
    for n, i in enumerate(arr):
        if n + 1 < len(arr):
            tok1 = arr[n]
            tok2 = arr[n + 1]
            if not is_paren(tok1) and not is_paren(tok2):
                arr[n + 1] = ""

    nonleaves = " ".join(arr)
    return nonleaves.split()
def dec(ids):
    tokens = tokenizer2.convert_ids_to_tokens(ids)
    text = "".join(tokens).replace("Ġ"," ")
    return text
'''
nlp = StanfordCoreNLP(r'/data/luoyt/stanford-corenlp-4.0.0', memory='8g', timeout=50000, quiet=False)

str1=nlp.parse("it 's what 1 -rrb- explains why we are like , well , ourselves otherwise than <UNK> jackson ; 2 -rrb- cautions that it 's possible to <UNK> in a lake that averages two feet deep ; and 3 -rrb- predicts that 10,000 <UNK> placed before 10,000 <UNK> would produce <UNK> <UNK> rock 'n' roll <UNK> .")
print(deleaf(str1))
'''
#再把deleaf的结果当作句法树输入 
cache_dir="/data/luoyt/dpattack/data/pretrained/bart-base"
bartconfig = BartConfig.from_pretrained('facebook/bart-base', cache_dir=cache_dir)
seq2seq = BartModel(bartconfig).to(device)#.from_pretrained('facebook/bart-base', cache_dir=cache_dir).to(device)
encoder = seq2seq.encoder
decoder = seq2seq.decoder
embed = seq2seq.get_input_embeddings()
ce = torch.nn.CrossEntropyLoss(reduction='mean')
mlp_word = torch.nn.Linear(bartconfig.d_model, bartconfig.vocab_size, bias=False).to(device)
seq2seq.train()
mlp_word.train()
optimizer = torch.optim.Adam(
            seq2seq.parameters(),
            lr=1e-3)
for i in range(1000):
    print("Epoch %d..." %i)
    with open('/data/luoyt/dpattack/data/tmp/ori.txt','r')as f:
        g = open('/data/luoyt/dpattack/data/tmp/adv.txt','r')
        ori_lines = f.readlines()
        adv_lines = g.readlines()
        for idx, line in enumerate(ori_lines):
            ids=list(map(int, line[:-1].split()))+[bartconfig.eos_token_id]
            dec_ids=list(map(int, adv_lines[idx][:-1].split()))+[bartconfig.eos_token_id]
            #ids=torch.tensor(ids).to(device)
            #dec_ids=torch.tensor(dec_ids).to(device)

            batch_ids=[ids]
            batch_dec_ids=[dec_ids]
            batch_ids = torch.tensor(batch_ids).to(device)#pad_sequence(batch_ids, batch_first=True, padding_value=1.0)
            batch_dec_ids = torch.tensor(batch_dec_ids).to(device)#pad_sequence(batch_dec_ids, batch_first=True, padding_value=1.0)
            src_mask = batch_ids[:,:-1] != tokenizer2.pad_token_id
            tgt_mask = batch_dec_ids != tokenizer2.pad_token_id
            enc_out=encoder(inputs_embeds=embed(batch_dec_ids),
                    attention_mask=tgt_mask,
                    return_dict=True)
            
            outputs = decoder(
                    inputs_embeds=embed(batch_ids[:,:-1]),
                    attention_mask=src_mask,
                    encoder_hidden_states=enc_out.hidden_states,
                    output_hidden_states=True,
                    return_dict=True)
            logits = mlp_word(outputs.last_hidden_state)

            preds=torch.argmax(logits,dim=-1)
            print("Gold: ", dec(batch_ids[0]), "\nPred: ", dec(preds[0]))
            labels = batch_ids[:, 1:]
            loss = ce(logits.view(-1,logits.size(-1)), labels.view(-1))
            print("loss: ",loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(mlp_word.parameters(), 5.0)
            
            optimizer.step()
            
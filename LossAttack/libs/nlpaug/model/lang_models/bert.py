# coding: utf-8
# Source: https://arxiv.org/abs/1810.04805
import os
try:
    import torch
    from transformers import BertTokenizer, BertForMaskedLM, BertConfig
    from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig
except ImportError:
    # No installation required if not using this function
    pass

from LossAttack.libs.nlpaug.model.lang_models import LanguageModels


class BertDeprecated(LanguageModels):
    START = '[CLS]'
    SEPARATOR = '[SEP]'
    MASK = '[MASK]'
    SUBWORD_PREFIX = '##'

    def __init__(self, model_path='bert-base-cased', tokenizer_path=None, device=None):
        super().__init__(device)
        self.model_path = model_path
        #bert_config = BertConfig.from_json_file(os.path.join(model_path, 'config.json'))
        if "bert-large" in model_path:
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-cased", cache_dir=model_path)
            self.model = BertForMaskedLM.from_pretrained("bert-large-cased", cache_dir=model_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", cache_dir=model_path)
            self.model = BertForMaskedLM.from_pretrained("bert-base-cased", cache_dir=model_path)
        self.model.to(device)
        self.model.eval()

    def predict(self, input_tokens, target_word, top_n):
        results = []
        tokens = input_tokens.copy()
        tokens.insert(0, BertDeprecated.START)
        tokens.append(BertDeprecated.SEPARATOR)

        # Mask target word
        target_pos = tokens.index(target_word)
        target_idx = self.tokenizer.convert_tokens_to_ids([target_word])[0]
        tokens[target_pos] = BertDeprecated.MASK

        # Convert feature
        token_idxes = self.tokenizer.convert_tokens_to_ids(tokens)
        segments_idxes = [0] * len(tokens)
        mask_idxes = [1] * len(tokens)  # 1: real token, 0: padding token

        tokens_features = torch.tensor([token_idxes]).to(self.device)
        segments_features = torch.tensor([segments_idxes]).to(self.device)
        mask_features = torch.tensor([mask_idxes]).to(self.device)

        # Predict target word
        outputs = self.model(tokens_features, segments_features, mask_features)
        logits = outputs[0]

        top_score_idx = target_idx
        for _ in range(100):
            logits[0, target_pos, top_score_idx] = -9999
            top_score_idx = torch.argmax(logits[0, target_pos]).item()
            top_score_token = self.tokenizer.convert_ids_to_tokens([top_score_idx])[0]
            if top_score_token[:2] != Bert.SUBWORD_PREFIX:
                results.append(top_score_token)
                if len(results) >= top_n:
                    break

        return results


class Bert(LanguageModels):
    START_TOKEN = '[CLS]'
    SEPARATOR_TOKEN = '[SEP]'
    MASK_TOKEN = '[MASK]'
    SUBWORD_PREFIX = '##'

    def __init__(self, model_path='bert-base-cased', temperature=1.0, top_k=None, top_p=None, device='cuda'):
        super().__init__(device, temperature=temperature, top_k=top_k, top_p=top_p)
        self.model_path = model_path
        if "bert-large" in model_path:
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-cased", cache_dir=model_path)
            self.model = BertForMaskedLM.from_pretrained("bert-large-cased", cache_dir=model_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", cache_dir=model_path)
            self.model = BertForMaskedLM.from_pretrained("bert-base-cased", cache_dir=model_path)
    
        self.model.to(self.device)
        self.model.eval()

    def id2token(self, _id):
        # id: integer format
        return self.tokenizer.convert_ids_to_tokens([_id])[0]

    def is_skip_candidate(self, candidate):
        return candidate[:2] == self.SUBWORD_PREFIX

    def predict(self, text, target_word=None, n=1):#双向语言模型，同时考虑MASK的前后文，因此需要完整句子。
        # Prepare inputs
        tokens = self.tokenizer.tokenize(text)
        tokens.insert(0, self.START_TOKEN)
        tokens.append(self.SEPARATOR_TOKEN)
        target_pos = tokens.index(self.MASK_TOKEN)

        token_inputs = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_inputs = [0] * len(token_inputs)
        mask_inputs = [1] * len(token_inputs)  # 1: real token, 0: padding token

        # Convert to feature
        token_inputs = torch.tensor([token_inputs]).to(self.device)
        segment_inputs = torch.tensor([segment_inputs]).to(self.device)
        mask_inputs = torch.tensor([mask_inputs]).to(self.device)

        # Prediction
        with torch.no_grad():
            outputs = self.model(token_inputs, segment_inputs, mask_inputs)
        target_token_logits = outputs[0][0][target_pos]

        # Selection
        seed = {'temperature': self.temperature, 'top_k': self.top_k, 'top_p': self.top_p}
        target_token_logits = self.control_randomness(target_token_logits, seed)
        target_token_logits, target_token_idxes = self.filtering(target_token_logits, seed)

        results = self.pick(target_token_logits, target_word=target_word, n=n)
        #print("res: ",results)#[('up', 0.23097165), ('substantial', 0.004066406), ('whole', 0.029276809), ('large', 0.29117775), ('major', 0.003024781), ('huge', 0.06783609), ('down', 0.014376994)]
        return results

class Roberta(LanguageModels):
    START_TOKEN = '<s>'
    SEPARATOR_TOKEN = '</s>'
    MASK_TOKEN = '<mask>'
    SUBWORD_PREFIX = 'Ġ'

    def __init__(self, model_path='roberta-base', temperature=1.0, top_k=None, top_p=None, device='cuda'):
        super().__init__(device, temperature=temperature, top_k=top_k, top_p=top_p)
        self.model_path = model_path
        print("init model Roberta...")
        if "roberta-large" in model_path:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large", cache_dir=model_path)
            self.model = RobertaForMaskedLM.from_pretrained("roberta-large", cache_dir=model_path)
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", cache_dir=model_path)
            self.model = RobertaForMaskedLM.from_pretrained("roberta-base", cache_dir=model_path)
    
        self.model.to(self.device)
        self.model.eval()

    def id2token(self, _id):
        # id: integer format
        return self.tokenizer.convert_ids_to_tokens([_id])[0]

    def is_skip_candidate(self, candidate):
        return candidate[:1] == self.SUBWORD_PREFIX

    def predict(self, text, target_word=None, n=1):#双向语言模型，同时考虑MASK的前后文，因此需要完整句子。
        # Prepare inputs
        tokens = self.tokenizer.tokenize(text)
        tokens.insert(0, self.START_TOKEN)
    
        target_pos = tokens.index(self.MASK_TOKEN)

        token_inputs = self.tokenizer.convert_tokens_to_ids(tokens)
        mask_inputs = [1] * len(token_inputs)  # 1: real token, 0: padding token

        # Convert to feature
        token_inputs = torch.tensor([token_inputs]).to(self.device)
        # Prediction
        with torch.no_grad():
            #这里报的错，看看啥情况
            print(token_inputs.shape)
            outputs = self.model(input_ids=token_inputs)
        target_token_logits = outputs[0][0][target_pos]

        # Selection
        seed = {'temperature': self.temperature, 'top_k': self.top_k, 'top_p': self.top_p}
        target_token_logits = self.control_randomness(target_token_logits, seed)
        target_token_logits, target_token_idxes = self.filtering(target_token_logits, seed)

        results = self.pick(target_token_logits, target_word=target_word, n=n)
        #print("res: ",results)#[('up', 0.23097165), ('substantial', 0.004066406), ('whole', 0.029276809), ('large', 0.29117775), ('major', 0.003024781), ('huge', 0.06783609), ('down', 0.014376994)]
        return results
# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from LossAttack.utils.metric import ParserMetric, TaggerMetric
from LossAttack.utils.parser_helper import is_chars_judger
from LossAttack.libs.luna import cast_list


class Task(object):
    def __init__(self, vocab, model):
        self.vocab = vocab
        self.model = model

    def train(self, loader, **kwargs):
        pass

    @torch.no_grad()
    def evaluate(self, loader, **kwargs):
        pass

    @torch.no_grad()
    def predict(self, loader, **kwargs):
        pass


class ParserTask(Task):
    def __init__(self, vocab, model):
        super(ParserTask, self).__init__(vocab, model)

        self.vocab = vocab
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def train(self, loader):
        self.model.train()

        for words, tags, chars, arcs, rels in loader:
            self.optimizer.zero_grad()

            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            # tags = self.get_tag(words, tags, mask)
            s_arc, s_rel = self.model(
                words, is_chars_judger(self.model, tags, chars))
            s_arc, s_rel = s_arc[mask], s_rel[mask]
            gold_arcs, gold_rels = arcs[mask], rels[mask]

            loss = self.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader, punct=False, tagger=None, mst=False):
        self.model.eval()

        loss, metric = 0, ParserMetric()

        for words, tags, chars, arcs, rels in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0

            tags = self.get_tags(words, tags, mask, tagger)
            
            s_arc, s_rel = self.model(
                words, is_chars_judger(self.model, tags, chars))

            loss += self.get_loss(s_arc[mask], s_rel[mask], arcs[mask], rels[mask])
            pred_arcs, pred_rels = self.decode(s_arc, s_rel, mask, mst)

            # ignore all punctuation if not specified
            if not punct:
                puncts = words.new_tensor(self.vocab.puncts)
                mask &= words.unsqueeze(-1).ne(puncts).all(-1)
            pred_arcs, pred_rels = pred_arcs[mask], pred_rels[mask]
            gold_arcs, gold_rels = arcs[mask], rels[mask]
            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)
        loss /= len(loader)

        return loss, metric

    # WARNING: DIRTY CODE >>>>>>>>>>>>>>>>>>>>>>>>>>>
    @torch.no_grad()
    def partial_evaluate(self, instance: tuple,
                         mask_idxs: List[int],
                         punct=False, tagger=None, mst=False, 
                         return_metric=True):
        self.model.eval()

        loss, metric = 0, ParserMetric()

        words, tags, chars, arcs, rels = instance

        mask = words.ne(self.vocab.pad_index)
        # ignore the first token of each sentence
        mask[:, 0] = 0
        decode_mask = mask.clone()

        tags = self.get_tags(words, tags, mask, tagger)
        # ignore all punctuation if not specified
        if not punct:
            puncts = words.new_tensor(self.vocab.puncts)
            mask &= words.unsqueeze(-1).ne(puncts).all(-1)
        s_arc, s_rel = self.model(
            words, is_chars_judger(self.model, tags, chars))

        # mask given indices
        for idx in mask_idxs:
            mask[:, idx] = 0

        pred_arcs, pred_rels = self.decode(s_arc, s_rel, decode_mask, mst)

        # punct is ignored !!!
        pred_arcs, pred_rels = pred_arcs[mask], pred_rels[mask]
        gold_arcs, gold_rels = arcs[mask], rels[mask]

        # exmask = torch.ones_like(gold_arcs, dtype=torch.uint8)

        # for i, ele in enumerate(cast_list(gold_arcs)):
        #     if ele in mask_idxs:
        #         exmask[i] = 0
        # for i, ele in enumerate(cast_list(pred_arcs)):
        #     if ele in mask_idxs:
        #         exmask[i] = 0
        # gold_arcs = gold_arcs[exmask]
        # pred_arcs = pred_arcs[exmask]
        # gold_rels = gold_rels[exmask]
        # pred_rels = pred_rels[exmask]

        # loss += self.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
        metric(pred_arcs, pred_rels, gold_arcs, gold_rels)

        if return_metric:
            return metric
        else:
            return pred_arcs.view(words.size(0), -1), pred_rels.view(words.size(0), -1), \
                   gold_arcs.view(words.size(0), -1), gold_rels.view(words.size(0), -1)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    @torch.no_grad()
    def predict(self, loader, tagger=None, mst=False):
        self.model.eval()

        all_tags, all_arcs, all_rels = [], [], []
        for words, tags, chars in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()

            tags = self.get_tags(words, tags, mask, tagger)
            s_arc, s_rel = self.model(
                words, is_chars_judger(self.model, tags, chars))

            pred_arcs, pred_rels = self.decode(s_arc, s_rel, mask, mst)
            tags, pred_arcs, pred_rels = tags[mask], pred_arcs[mask], pred_rels[mask]
            

            all_tags.extend(torch.split(tags, lens))
            all_arcs.extend(torch.split(pred_arcs, lens))
            all_rels.extend(torch.split(pred_rels, lens))
        all_tags = [self.vocab.id2tag(seq) for seq in all_tags]
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_tags, all_arcs, all_rels

    def get_loss(self, s_arc, s_rel, gold_arcs, gold_rels):
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs]
        # s_rel = s_rel[torch.arange(len(gold_arcs)), gold_arcs]

        arc_loss = self.criterion(s_arc, gold_arcs)
        rel_loss = self.criterion(s_rel, gold_rels)
        loss = arc_loss + rel_loss

        return loss

    def decode(self, s_arc, s_rel, mask, mst):
        from LossAttack.utils.alg import eisner
        if mst:
            pred_arcs = eisner(s_arc, mask)
        else:
            pred_arcs = s_arc.argmax(dim = -1)
        # pred_arcs = s_arc.argmax(dim=-1)
        # pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)
        pred_rels = s_rel.argmax(-1)
        pred_rels = pred_rels.gather(-1, pred_arcs.unsqueeze(-1)).squeeze(-1)

        return pred_arcs, pred_rels

    def get_tags(self, words, tags, mask, tagger):
        if tagger is None:
            return tags
        else:
            tagger = tagger.eval()
            lens = mask.sum(dim=1).tolist()
            s_tags = tagger(words)
            pred_tags = s_tags[mask].argmax(-1)
            pred_tags = torch.split(pred_tags, lens)
            pred_tags = pad_sequence(pred_tags, True)
            pred_tags = torch.cat(
                [torch.zeros_like(pred_tags[:, :1]), pred_tags], dim=1)
            return pred_tags

class StackPtrParserTask(Task):
    def __init__(self, vocab, model):
        super(StackPtrParserTask, self).__init__(vocab, model)

        self.vocab = vocab
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def train(self, loader):
        self.model.train()
        step=0

        for words, tags, chars, arcs, rels in loader:
            # shape: [bsz, seq_len]
            self.optimizer.zero_grad()

            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            # tags = self.get_tag(words, tags, mask)
            loss_arc, loss_rel = self.model.loss(words, chars, tags, arcs, rels)
            loss = torch.sum(loss_arc + loss_rel)/words.shape[0]
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.scheduler.step()
            if step % 10==0:
                print("Step {}, bloss: {}.".format(step, loss.item()/loss_arc.shape[0]))
            step+=1

    @torch.no_grad()
    def evaluate(self, loader, punct=False, tagger=None, mst=False):
        self.model.eval()

        loss, metric = 0, ParserMetric()

        for words, tags, chars, arcs, rels in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0

            tags = self.get_tags(words, tags, mask, tagger)
            
            pred_arcs, pred_rels  = self.model.decode(
                words,  chars, tags, mask=mask)

            # ignore all punctuation if not specified
            if not punct:
                puncts = words.new_tensor(self.vocab.puncts)
                mask &= words.unsqueeze(-1).ne(puncts).all(-1)
            pred_arcs, pred_rels = pred_arcs[mask.cpu()], pred_rels[mask.cpu()]
            gold_arcs, gold_rels = arcs[mask.cpu()], rels[mask.cpu()]

            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)
        loss /= len(loader)

        return loss, metric

    
    @torch.no_grad()
    def predict(self, loader, tagger=None, mst=False):
        self.model.eval()

        all_tags, all_arcs, all_rels = [], [], []
        for words, tags, chars in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()

            tags = self.get_tags(words, tags, mask, tagger)
            pred_arcs, pred_rels = self.model.decode(
                words,  chars, tags, mask=mask)

            tags, pred_arcs, pred_rels = tags[mask.cpu()], pred_arcs[mask.cpu()], pred_rels[mask.cpu()]
            

            all_tags.extend(torch.split(tags, lens))
            all_arcs.extend(torch.split(pred_arcs, lens))
            all_rels.extend(torch.split(pred_rels, lens))
        all_tags = [self.vocab.id2tag(seq) for seq in all_tags]
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_tags, all_arcs, all_rels

    def get_loss(self, s_arc, s_rel, gold_arcs, gold_rels):
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs]
        # s_rel = s_rel[torch.arange(len(gold_arcs)), gold_arcs]

        arc_loss = self.criterion(s_arc, gold_arcs)
        rel_loss = self.criterion(s_rel, gold_rels)
        loss = arc_loss + rel_loss

        return loss

    def decode(self, s_arc, s_rel, mask, mst):
        from LossAttack.utils.alg import eisner
        if mst:
            pred_arcs = eisner(s_arc, mask)
        else:
            pred_arcs = s_arc.argmax(dim = -1)
        # pred_arcs = s_arc.argmax(dim=-1)
        # pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)
        pred_rels = s_rel.argmax(-1)
        pred_rels = pred_rels.gather(-1, pred_arcs.unsqueeze(-1)).squeeze(-1)

        return pred_arcs, pred_rels

    def get_tags(self, words, tags, mask, tagger):
        if tagger is None:
            return tags
        else:
            tagger = tagger.eval()
            lens = mask.sum(dim=1).tolist()
            s_tags = tagger(words)
            pred_tags = s_tags[mask].argmax(-1)
            pred_tags = torch.split(pred_tags, lens)
            pred_tags = pad_sequence(pred_tags, True)
            pred_tags = torch.cat(
                [torch.zeros_like(pred_tags[:, :1]), pred_tags], dim=1)
            return pred_tags



class TaggerTask(Task):
    def __init__(self, vocab, model):
        super(TaggerTask, self).__init__(vocab, model)

        self.vocab = vocab
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def train(self, loader, **kwargs):
        self.model.train()
        loss, metric = 0, TaggerMetric()
        for words, tags, chars, arcs, rels in loader:
            self.optimizer.zero_grad()
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_tags = self.model(words)
            s_tags = s_tags[mask]
            gold_tags = tags[mask]

            loss = self.get_loss(s_tags, gold_tags)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.scheduler.step()
            pred_tags = self.decode(s_tags)

            loss += self.get_loss(s_tags, gold_tags)
            metric(pred_tags, gold_tags)
        loss /= len(loader)

        return loss, metric

    @torch.no_grad()
    def evaluate(self, loader, punct=False, **kwargs):
        self.model.eval()

        loss, metric = 0, TaggerMetric()

        for words, tags, chars, arcs, rels in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            # ignore all punctuation if not specified
            s_tags = self.model(words)
            s_tags = s_tags[mask]
            gold_tags = tags[mask]
            pred_tags = self.decode(s_tags)

            loss += self.get_loss(s_tags, gold_tags)
            metric(pred_tags, gold_tags)
        loss /= len(loader)

        return loss, metric

    @torch.no_grad()
    def predict(self, loader, **kwargs):
        self.model.eval()

        all_tags = []
        for words, tags, chars in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()
            s_tags = self.model(words)
            s_tags = s_tags[mask]
            pred_tags = self.decode(s_tags)

            all_tags.extend(torch.split(pred_tags, lens))
        all_tags = [self.vocab.id2tag(seq) for seq in all_tags]
        return all_tags

    def get_loss(self, s_tags, gold_tags):
        loss = self.criterion(s_tags, gold_tags)
        return loss

    def decode(self, s_tags):
        pred_tags = s_tags.argmax(dim=-1)
        return pred_tags

class DECTask(Task):#TODO: ?????????????????????task????????????loader?????????????????????feed??????
    def __init__(self, vocab, model):
        super(DECTask, self).__init__(vocab, model)

        self.vocab = vocab
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocab size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            # torch.topk()???????????????????????????top_k??????????????????????????????(values,indices)
            # ...??????????????????????????????????????????
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value  # ??????topk????????????????????????logits??????????????????

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # ???logits??????????????????
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def acc_sent(self, logits, labels):
        cnt=0
        corr=0
        _, logits = logits.max(dim=-1)
        for idx, label in enumerate(labels):
            cnt+=label.size(0)
            for index in label:
                if index in logits[idx]:
                    corr+=1
        return corr/cnt

    def calculate_loss_and_accuracy(self, logits, labels, device):
        """
        ?????????pad_id?????????loss????????????
        :param logits:
        :param labels:
        :param device:
        :return:
        """
        # ??????token?????????????????????token???prediction_score,??????:[batch_size,token_len,vocab_size]
        # ??????n-1???token???????????????n???token
        # ??????i???token???prediction_score???????????????i+1???token???
        # ?????????input???n???token??????shift_logits??????model??????[0,n-2]???token???prediction_score???shift_labels?????????[1???n-1]???label


        loss_fct = nn.CrossEntropyLoss(ignore_index=self.model.pad_token_id, reduction='sum')  # ??????pad_id???loss,??????????????????pad_id???loss????????????
        loss = loss_fct(logits.view(-1, logits.size(-1)),
                        labels.view(-1))

        _, preds = logits.max(dim=-1)  # preds???????????????prediction_score????????????token???vocab??????id????????????[batch_size,token_len]

        # ??????pad_id???token???loss????????????????????????????????????????????????
        not_ignore = labels.ne(self.model.pad_token_id)  # ??????????????????????????????tensor??????targets_view??????i????????????pad_id????????????0????????????1
        num_targets = not_ignore.long().sum().item()  # ??????target?????????pad_id?????????

        correct = (labels == preds) & not_ignore  # ??????model???????????????token??????????????????pad???token
        correct = correct.float().sum()

        accuracy = correct / num_targets
        loss = loss / num_targets
        return loss, accuracy

    def pad_sequence(sequences, batch_first=False, padding_value=0.0):
        # type: (List[tensor], bool, float) -> tensor
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
            padding_cnt.append(max_len-length)#padding??????
            # use index notation to prevent duplicate references to the tensor
            if batch_first:
                out_tensor[i, :length, ...] = tensor
            else:
                out_tensor[:length, i, ...] = tensor

        return out_tensor, padding_cnt

    def train(self, loader):
        print("Training Phase...")
        self.model.train()
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.model.pad_token_id, reduction='sum')
        loss, metric = 0, ParserMetric()# UAS & LAS
        pbar = tqdm(loader)
        for words, tags, arcs, rels, tree, sub_len, atk_words, atk_tags, atk_arcs, atk_rels, atk_tree, atk_sub_len in pbar:
            #print("model input shape for words, tags, arcs, rels,  sub_len, atk_words, atk_tags, atk_arcs, atk_rels, atk_sub_len: ", 
            #words.shape, tags.shape, arcs.shape, rels.shape, sub_len.shape, atk_words.shape, atk_tags.shape, atk_arcs.shape, atk_rels.shape, atk_sub_len.shape)
            #words: [bsz, seq_len, embed_dim]
            self.optimizer.zero_grad()
            logits_out, logits_arcs, logits_rels, labels, labels_arcs, labels_rels = self.model(
                                atk_words, atk_tags, atk_arcs, atk_rels, atk_tree, words, tags, arcs, rels, tree)
            
            #print("model output shape for logits_out, logits_arcs, logits_rels, labels, labels_arcs, labels_rels: ",
            #logits_out.shape, logits_arcs.shape, logits_rels.shape, labels.shape, labels_arcs.shape, labels_rels.shape)
            pred_words = torch.argmax(logits_out,dim=-1)
            pred_arcs = torch.argmax(logits_arcs,dim=-1)
            pred_rels = torch.argmax(logits_rels,dim=-1)
            
            loss1, acc1 = self.calculate_loss_and_accuracy(logits_out, labels, device=self.model.cuda_device)
            #???????????????arc rel???loss???????????????????????????????????????????????????????????????????????????Parser??????????????????????????????
            m_loss = loss1
            
            acc = self.acc_sent(logits_out, labels)
            m_loss.backward()
            #argmax decode?????????????????????the?????????????????????????????????nucleus sampling
            print("Gold Sent {}: ".format(0), self.vocab.id2word(words[0]))
            print("Pred Sent {}: ".format(0), self.vocab.id2word(pred_words[0]))
            loss+=m_loss.item()
            pbar.set_description("M_loss: {:.3f}, accuracy: {:.3f}".format(m_loss.item(), acc))
            #pbar.set_description("M_loss: {:.3f}; loss1&2&3: {:.3f}, {:.3f}, {:.3f}, accuracy: {:.3f}".format(m_loss.item(), loss1.item(), loss2.item(), loss3.item(), acc))
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.scheduler.step()
            
            #check ???words???sub_len??????????????????sub_words???sub_len????????????
            #collate_fn?????????
            #print("-----------------------------------------")
            #for idx,sublen in enumerate(sub_len):
                #print(torch.sum(sublen).item(), words[idx].size(0))
                #assert torch.sum(sublen).item()==words[idx].size(0)
            #print("=========================================")
            #for idx,atksublen in enumerate(atk_sub_len):
                #print(torch.sum(atksublen).item(), atk_words[idx].size(0))
                #assert torch.sum(atksublen).item()==atk_words[idx].size(0),print(atksublen, atk_words[idx])
            
            

            #TODO: ??????????????????topktopp
            
            '''
            # ignore all punctuation if not specified
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            pred_arcs, pred_rels = pred_arcs[mask], pred_rels[mask]
            gold_arcs, gold_rels = arcs[mask], rels[mask]

            #metric(pred_arcs, pred_rels, gold_arcs, gold_rels)
            '''
        loss /= len(loader)

        return loss, metric

    @torch.no_grad()
    def evaluate(self, loader, punct=False, tagger=None, mst=False):
        print("Evaluating Phase...")
        self.model.eval()
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.model.pad_token_id, reduction='sum')
        loss, metric = 0, ParserMetric()
        pbar = tqdm(loader)
        for words, tags, arcs, rels, tree, sub_len, atk_words, atk_tags, atk_arcs, atk_rels, atk_tree, atk_sub_len in pbar:
            #print("model input shape for words, tags, arcs, rels,  sub_len, atk_words, atk_tags, atk_arcs, atk_rels, atk_sub_len: ", 
            #words.shape, tags.shape, arcs.shape, rels.shape, sub_len.shape, atk_words.shape, atk_tags.shape, atk_arcs.shape, atk_rels.shape, atk_sub_len.shape)
            #words: [bsz, seq_len, embed_dim]

            logits_out, logits_arcs, logits_rels, labels, labels_arcs, labels_rels = self.model(
                                atk_words, atk_tags, atk_arcs, atk_rels, atk_tree, words, tags, arcs, rels, tree)
            
            #print("model output shape for logits_out, logits_arcs, logits_rels, labels, labels_arcs, labels_rels: ",
            #logits_out.shape, logits_arcs.shape, logits_rels.shape, labels.shape, labels_arcs.shape, labels_rels.shape)

            pred_words = torch.argmax(logits_out,dim=-1)
            pred_arcs = torch.argmax(logits_arcs,dim=-1)
            pred_rels = torch.argmax(logits_rels,dim=-1)
            '''
            #???label???logits??????            
            #???????????????label??????
            gold_arcs=[[]]
            gold_rels=[[]]
            gold_words=[[]]
            for idx, sublen in enumerate(sub_len):
                cnt=0
                for j in sublen:
                    if cnt<words[idx].size(0):
                        gold_arcs[-1].append(arcs[idx, cnt])#??????subwords????????????label????????????word???????????????
                        gold_rels[-1].append(rels[idx, cnt])#??????subwords????????????label????????????word???????????????
                        gold_words[-1].append(words[idx, cnt])#??????subwords????????????token????????????word???????????????
                        cnt+=j
                #????????????????????????
                gold_arcs.append([])
                gold_rels.append([])
                gold_words.append([])

            del gold_arcs[-1]
            del gold_rels[-1]
            del gold_words[-1]
                
            for idx in range(len(gold_arcs)):
                gold_arcs[idx] = torch.tensor(gold_arcs[idx],dtype=arcs.dtype).to(self.model.cuda_device)
            arcs = pad_sequence(gold_arcs,True)
            for idx in range(len(gold_rels)):
                gold_rels[idx] = torch.tensor(gold_rels[idx],dtype=rels.dtype).to(self.model.cuda_device)
            rels = pad_sequence(gold_rels,True)
            for idx in range(len(gold_words)):
                gold_words[idx] = torch.tensor(gold_words[idx],dtype=words.dtype).to(self.model.cuda_device)
            words = pad_sequence(gold_words,True)
            #print("resize gold shape: ",words.shape, arcs.shape, rels.shape)

            #??????????????????
            #?????????????????????sublen???pred_arcs
            pred_newarcs = [[]]
            pred_newrels = [[]]
            pred_newwords=[[]]
            for idx, sublen in enumerate(sub_len):
                cnt=0
                for j in sublen:
                    if cnt<pred_words[idx].size(0):
                        pred_newarcs[-1].append(pred_arcs[idx, cnt])#??????subwords????????????label????????????word???????????????
                        pred_newrels[-1].append(pred_rels[idx, cnt])#??????subwords????????????label????????????word???????????????
                        pred_newwords[-1].append(pred_words[idx, cnt])#??????subwords????????????token????????????word???????????????
                        cnt+=j
                #????????????????????????
                pred_newarcs.append([])
                pred_newrels.append([])
                pred_newwords.append([])
            del pred_newarcs[-1]
            del pred_newrels[-1]
            del pred_newwords[-1]
                
            for idx in range(len(pred_newarcs)):
                pred_newarcs[idx] = torch.tensor(pred_newarcs[idx],dtype=pred_arcs.dtype).to(self.model.cuda_device)
            pred_arcs = pad_sequence(pred_newarcs,True)
            for idx in range(len(pred_newrels)):
                pred_newrels[idx] = torch.tensor(pred_newrels[idx],dtype=pred_rels.dtype).to(self.model.cuda_device)
            pred_rels = pad_sequence(pred_newrels,True)
            for idx in range(len(pred_newwords)):
                pred_newwords[idx] = torch.tensor(pred_newwords[idx],dtype=pred_words.dtype).to(self.model.cuda_device)
            pred_words = pad_sequence(pred_newwords,True)
            #print("resize prediction shape: ", pred_words.shape, pred_arcs.shape, pred_rels.shape)
            loss1, acc1 = self.calculate_loss_and_accuracy(logits_out, labels, device=self.model.cuda_device)
            loss2, acc2 = self.calculate_loss_and_accuracy(logits_arcs, labels_arcs, device=self.model.cuda_device) 
            loss3, acc3 = self.calculate_loss_and_accuracy(logits_rels, labels_rels, device=self.model.cuda_device)
            m_loss = loss1+loss2+loss3
            acc = (acc2+acc3)/2.0
            #argmax decode?????????????????????the?????????????????????????????????nucleus sampling
            loss+=m_loss.item()
            pbar.set_description("M_loss: {:.3f}; loss1&2&3: {:.3f}, {:.3f}, {:.3f}, accuracy: {:.3f}".format(m_loss.item(), loss1.item(), loss2.item(), loss3.item(), acc))
            '''
            loss1, acc1 = self.calculate_loss_and_accuracy(logits_out, labels, device=self.model.cuda_device)
            #???????????????arc rel???loss???????????????????????????????????????????????????????????????????????????Parser??????????????????????????????
            m_loss = loss1
            acc = self.acc_sent(logits_out, labels)
            loss+=m_loss.item()
            pbar.set_description("M_loss: {:.3f}, accuracy: {:.3f}".format(m_loss.item(), acc))
            
        loss /= len(loader)

        return loss, metric

    # WARNING: DIRTY CODE >>>>>>>>>>>>>>>>>>>>>>>>>>>
    @torch.no_grad()
    def partial_evaluate(self, instance: tuple,
                         mask_idxs: List[int],
                         punct=False, tagger=None, mst=False, 
                         return_metric=True):
        self.model.eval()

        loss, metric = 0, ParserMetric()

        words, tags, chars, arcs, rels = instance

        mask = words.ne(self.vocab.pad_index)
        # ignore the first token of each sentence
        mask[:, 0] = 0
        decode_mask = mask.clone()

        tags = self.get_tags(words, tags, mask, tagger)
        # ignore all punctuation if not specified
        if not punct:
            puncts = words.new_tensor(self.vocab.puncts)
            mask &= words.unsqueeze(-1).ne(puncts).all(-1)
        s_arc, s_rel = self.model(
            words, is_chars_judger(self.model, tags, chars))

        # mask given indices
        for idx in mask_idxs:
            mask[:, idx] = 0

        pred_arcs, pred_rels = self.decode(s_arc, s_rel, decode_mask, mst)

        # punct is ignored !!!
        pred_arcs, pred_rels = pred_arcs[mask], pred_rels[mask]
        gold_arcs, gold_rels = arcs[mask], rels[mask]

        # exmask = torch.ones_like(gold_arcs, dtype=torch.uint8)

        # for i, ele in enumerate(cast_list(gold_arcs)):
        #     if ele in mask_idxs:
        #         exmask[i] = 0
        # for i, ele in enumerate(cast_list(pred_arcs)):
        #     if ele in mask_idxs:
        #         exmask[i] = 0
        # gold_arcs = gold_arcs[exmask]
        # pred_arcs = pred_arcs[exmask]
        # gold_rels = gold_rels[exmask]
        # pred_rels = pred_rels[exmask]

        # loss += self.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
        metric(pred_arcs, pred_rels, gold_arcs, gold_rels)

        if return_metric:
            return metric
        else:
            return pred_arcs.view(words.size(0), -1), pred_rels.view(words.size(0), -1), \
                   gold_arcs.view(words.size(0), -1), gold_rels.view(words.size(0), -1)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    @torch.no_grad()
    def predict(self, loader, tagger=None, mst=False):
        self.model.eval()

        all_tags, all_arcs, all_rels = [], [], []
        for words, tags, chars in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()

            tags = self.get_tags(words, tags, mask, tagger)
            s_arc, s_rel = self.model(
                words, is_chars_judger(self.model, tags, chars))

            pred_arcs, pred_rels = self.decode(s_arc, s_rel, mask, mst)
            tags, pred_arcs, pred_rels = tags[mask], pred_arcs[mask], pred_rels[mask]
            

            all_tags.extend(torch.split(tags, lens))
            all_arcs.extend(torch.split(pred_arcs, lens))
            all_rels.extend(torch.split(pred_rels, lens))
        all_tags = [self.vocab.id2tag(seq) for seq in all_tags]
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_tags, all_arcs, all_rels

    def get_loss(self, s_arc, s_rel, gold_arcs, gold_rels):
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs]
        # s_rel = s_rel[torch.arange(len(gold_arcs)), gold_arcs]

        arc_loss = self.criterion(s_arc, gold_arcs)
        rel_loss = self.criterion(s_rel, gold_rels)
        loss = arc_loss + rel_loss

        return loss

    def decode(self, s_arc, s_rel, mask, mst):
        from LossAttack.utils.alg import eisner
        if mst:
            pred_arcs = eisner(s_arc, mask)
        else:
            pred_arcs = s_arc.argmax(dim = -1)
        # pred_arcs = s_arc.argmax(dim=-1)
        # pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)
        pred_rels = s_rel.argmax(-1)
        pred_rels = pred_rels.gather(-1, pred_arcs.unsqueeze(-1)).squeeze(-1)

        return pred_arcs, pred_rels

    def get_tags(self, words, tags, mask, tagger):
        if tagger is None:
            return tags
        else:
            tagger = tagger.eval()
            lens = mask.sum(dim=1).tolist()
            s_tags = tagger(words)
            pred_tags = s_tags[mask].argmax(-1)
            pred_tags = torch.split(pred_tags, lens)
            pred_tags = pad_sequence(pred_tags, True)
            pred_tags = torch.cat(
                [torch.zeros_like(pred_tags[:, :1]), pred_tags], dim=1)
            return pred_tags

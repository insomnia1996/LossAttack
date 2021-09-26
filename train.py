import argparse
import time,os
import torch
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import pickle
from transformers import BartTokenizer
from tqdm import tqdm

def null(model, opt, arg):# save count_map {sentid_in_train: sentid} to match each sentence.
    print("fetching number map...")
    f  = open(os.path.join(opt.data_path, "result","count_map.txt"), 'w')
    if arg=='train':
        loader = opt.train
    elif arg=='eval':
        loader = opt.eval
    else:
        loader = opt.test
    for i, batch in enumerate(loader): 
        trg, trg_arc, trg_rel, src, src_arc, src_rel, number = list(batch)
        for n in number:
            #print(n)
            f.write(str(n.item())+'\n')
    f.close()

def train_model(model, opt, arg):
    tokenizer2 = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir = "./data/pretrained/bart-base")
    print("training model...")
    model.train()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()
    best=0
    if arg=='train':
        loader = opt.train
    elif arg=='eval':
        loader = opt.eval
    else:
        loader = opt.test
    for epoch in range(opt.epochs):

        total_loss = 0
        if opt.floyd is False:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        
        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')
                    
        for i, batch in enumerate(loader): 
            #src = batch.src.transpose(0,1)
            #trg = batch.trg.transpose(0,1)
            trg, trg_arc, trg_rel, src, src_arc, src_rel, _ = list(batch)# ori_sent, ori_arc, ori_rel, adv_sent, adv_arc, adv_rel
            trg_input = trg[:, :-1]
            trg_arc = trg_arc[:,:-1]#长度与trg_sent不等，需单独mask
            trg_rel = trg_rel[:,:-1]#长度与trg_sent不等，需单独mask
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            #print("input shape: ", src.shape, trg_input.shape, src_arc.shape, trg_arc.shape)
            #print("mask shape: ", src_mask.shape, trg_mask.shape, src_anr_mask.shape, trg_anr_mask.shape)
            #print(src_mask, trg_mask)#enc mask全1，dec mask为下三角阵
            preds = model(src, src_arc, src_rel, trg_input, src_mask, trg_mask)
            #preds = model(src, trg_input, src_mask, trg_mask)#不输入ground true句法树，因为作为denoiser获取不到

            opt.optimizer.zero_grad()
            ys = trg[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=opt.trg_pad, reduction='mean')
            loss = loss_fct(preds.view(-1, preds.size(-1)), ys.view(-1))
            acc = acc_sent(preds, ys, ignore_index=opt.trg_pad)
            #print("ori: ", tokenizer2.decode(src[0]))
            #print("gold: ", tokenizer2.decode(trg[0]))
            print("pred: ", tokenizer2.decode(torch.argmax(preds,-1)[0]))
            print("loss: %.4f, acc: %.4f. " % (loss.item(), acc))
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True:
                opt.sched.step()
            
            total_loss += loss.item()
            
            if (i + 1) % opt.printevery == 0:
                 p = int(100 * (i + 1) / opt.train_len)
                 avg_loss = total_loss/opt.printevery
                 if opt.floyd is False:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f  acc = %.4f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss, acc), end='\r')
                 else:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f  acc = %.4f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss, acc))
                 total_loss = 0
                


        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f  acc = %.4f" %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss, acc))
        if epoch > opt.patience and acc > best:
            best=acc
            torch.save(model.state_dict(), 'weight/model_weights.pkl')#_denoiser
        evaluate_model(model, opt, 'eval')


def evaluate_model(model, opt, arg):
    #state_dict = torch.load('weight/model_weights.pkl')
    #model.load_state_dict(state_dict)

    tokenizer2 = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir = "./data/pretrained/bart-base")
    print("evaluating...")
    model.eval()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()
    total_loss = 0    
    if arg=='train':
        loader = opt.train
    elif arg=='eval':
        loader = opt.eval
    else:
        loader = opt.test  
    for i, batch in enumerate(loader): 
        #MODIFIED
        #src = batch.src.transpose(0,1)
        #trg = batch.trg.transpose(0,1)
        trg, trg_arc, trg_rel, src, src_arc, src_rel, _ = list(batch)# ori_sent, ori_arc, ori_rel, adv_sent, adv_arc, adv_rel
        trg_input = trg[:, :-1]
        trg_arc = trg_arc[:,:-1]#长度与trg_sent不等，需单独mask
        trg_rel = trg_rel[:,:-1]#长度与trg_sent不等，需单独mask
        src_mask, trg_mask = create_masks(src, trg_input, opt)
        #print("input shape: ", src.shape, trg_input.shape, src_arc.shape, trg_arc.shape)
        #print("mask shape: ", src_mask.shape, trg_mask.shape, src_anr_mask.shape, trg_anr_mask.shape)
        #print(src_mask, trg_mask)#enc mask全1，dec mask为下三角阵
        preds = model(src, src_arc, src_rel, trg_input, src_mask, trg_mask)
        #preds = model(src, trg_input, src_mask, trg_mask)#不输入ground true句法树，因为作为denoiser获取不到

        ys = trg[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=opt.trg_pad, reduction='mean')
        loss = loss_fct(preds.view(-1, preds.size(-1)), ys.view(-1))
        acc = acc_sent(preds, ys, ignore_index=opt.trg_pad)
        #print("gold: ", tokenizer2.decode(trg[0]))
        #print("pred: ", tokenizer2.decode(torch.argmax(preds,-1)[0]))
        #print("loss: %.4f, acc: %.4f. " % (loss.item(), acc))

        
        total_loss += loss.item()
        
        if (i + 1) % opt.printevery == 0:
            p = int(100 * (i + 1) / opt.train_len)
            avg_loss = total_loss/opt.printevery
            if opt.floyd is False:
                print("   %dm: [%s%s]  %d%%  loss = %.3f  acc = %.4f" %\
                ((time.time() - start)//60, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss, acc), end='\r')
            else:
                print("   %dm: [%s%s]  %d%%  loss = %.3f  acc = %.4f" %\
                ((time.time() - start)//60, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss, acc))
                total_loss = 0

def predict(model, opt, arg):
    state_dict = torch.load('weight/model_weights.pkl')
    model.load_state_dict(state_dict)
    f = open(os.path.join(opt.data_path, "result","res.txt"), 'w')
    g = open(os.path.join(opt.data_path, "result","gold.txt"), 'w')
    h = open(os.path.join(opt.data_path, "result","goldarc.txt"), 'w')
    k = open(os.path.join(opt.data_path, "result","goldrel.txt"), 'w')
    tokenizer2 = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir = "./data/pretrained/bart-base")
    print("predicting...")
    model.eval()
    start = time.time()
    cnt=0
    ttl=0
    if arg=='train':
        loader = opt.train
    elif arg=='eval':
        loader = opt.eval
    else:
        loader = opt.test
    for i, batch in enumerate(tqdm(loader)): 
        trg, trg_arc, trg_rel, src, src_arc, src_rel, _ = list(batch)# ori_sent, ori_arc, ori_rel, adv_sent, adv_arc, adv_rel
        
        trg_input = trg[:, :-1]
        trg_arc = trg_arc[:,:-1]#长度与trg_sent不等，需单独mask
        trg_rel = trg_rel[:,:-1]#长度与trg_sent不等，需单独mask
        ys = trg[:, 1:].contiguous()
        true_arc = trg_arc[:, 1:].contiguous()
        true_rel = trg_rel[:, 1:].contiguous()
        for j in range(src.size(0)):
            pred = model.predict(src[j:j+1,:], src_arc[j:j+1,:], src_rel[j:j+1,:], opt)#trg[j:j+1,:],
            #pred = model(src[j:j+1,:], opt)#不输入ground true句法树，因为作为denoiser获取不到
            acc = acc_recon(pred.unsqueeze(0), ys[j:j+1], tokenizer2)
            f.write(tokenizer2.decode(pred,clean_up_tokenization_spaces=False).split("</s>")[0]+'\n')
            g.write(tokenizer2.decode(ys[j],clean_up_tokenization_spaces=False).split("</s>")[0]+'\n')
            h.write(rm_pad(true_arc[j])+'\n')
            k.write(rm_pad(true_rel[j])+'\n')
            
            if acc==1:
                cnt+=1
            ttl+=1
        if (i + 1) % opt.printevery == 0:
            p = int(100 * (i + 1) / opt.train_len)
            if opt.floyd is False:
                print("   %dm: [%s%s]  %d%%  acc = %.4f" %\
                ((time.time() - start)//60, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, cnt/ttl), end='\r')
            else:
                print("   %dm: [%s%s]  %d%%  acc = %.4f" %\
                ((time.time() - start)//60, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, cnt/ttl))
    f.close()
    g.close()
    h.close()
    k.close()
    #before attack, dependency parser accuracy.

    #after attack

def rm_pad(lst, pad_id=1, eos_id=0):#tensor
    lst =  lst.tolist()
    n_pad = 0
    for k in lst[::-1]:
        
        if k == pad_id:
            n_pad += 1
        elif k == eos_id:
            n_pad += 1
            break
        else:
            break
            
    if n_pad!=0:
        lst = lst[:-n_pad]#留下句号对应label

    lst = list(map(str,lst))
    return " ".join(lst)

def acc_sent(logits, labels, ignore_index):#(bsz, seq_len, vocab)&(bsz, seq_len)
    bsz=logits.size(0)# default is 1
    assert logits.size(1)==labels.size(1)
    cnt=torch.sum(labels!=ignore_index)
    corr=0
    _, logits = logits.max(dim=-1)
    for i in range(bsz):
        for index in labels[i]:
            if index in logits[i] and index!=ignore_index:
                corr+=1
    return corr/cnt


def acc_recon(logits, labels, tokenizer, eos_id=2):
    assert len(logits.shape)==2#(bsz, seqlen1)
    assert len(labels.shape)==2#(bsz, seqlen2)
    
    bsz=labels.size(0)
    corr=0
    
    for idx in range(bsz):
        label = labels[idx]
        logit = logits[idx]
        label = tokenizer.decode(label,clean_up_tokenization_spaces=False).split("</s>")[0]
        logit = tokenizer.decode(logit,clean_up_tokenization_spaces=False).split("</s>")[0]
        #print("gold: ", label)
        #print("pred: ", logit)
        if label==logit:
            corr+=1
    return corr/bsz




def main_for_bi_tir():
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-patience', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-seed', type=int, default=1996)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=200)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-k', type=int, default=20)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=200)
    parser.add_argument('-is_train', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-device', type=int, default=0)
    parser.add_argument('-tensor_dir', type=str, default='/data/luoyt/dpattack/data/tensor')
    parser.add_argument('-data_path', type=str, default='/data/luoyt/dpattack/data')

    opt = parser.parse_args()
    
    opt.device = opt.device if opt.no_cuda is False else -1
    if opt.device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.device)
        assert torch.cuda.is_available()

    torch.manual_seed(opt.seed)


    if not os.listdir(opt.tensor_dir):
        os.mkdir(os.path.join(opt.tensor_dir,'train'))
        os.mkdir(os.path.join(opt.tensor_dir,'eval'))
        os.mkdir(os.path.join(opt.tensor_dir,'test'))
        opt.train = create_dataset(opt, os.path.join(opt.data_path, "result", "black_substitute_train_0.01.txt"),
                                    os.path.join(opt.data_path, "ptb", "ptb_train_3.3.0.sd"),
                                    os.path.join(opt.data_path, "result", "black_substitute_train_0.01.conllx"), train_or_dev='train')
        opt.eval = create_dataset(opt, os.path.join(opt.data_path, "result", "black_substitute_valid_0.01.txt"),
                                    os.path.join(opt.data_path, "ptb", "ptb_valid_3.3.0.sd"),
                                    os.path.join(opt.data_path, "result", "black_substitute_valid_0.01.conllx"), train_or_dev='eval')
        opt.test = create_dataset(opt, os.path.join(opt.data_path, "result", "black_substitute_test_0.01.txt"),
                                    os.path.join(opt.data_path, "ptb", "ptb_test_3.3.0.sd"),
                                    os.path.join(opt.data_path, "result", "black_substitute_test_0.01.conllx"), train_or_dev='test')
    else:
        opt.train = load_dataset(opt, os.path.join(opt.data_path, "result", "black_substitute_train_0.01.txt"),
                                    os.path.join(opt.data_path, "ptb", "ptb_train_3.3.0.sd"),
                                    os.path.join(opt.data_path, "result", "black_substitute_train_0.01.conllx"), train_or_dev='train')
        opt.eval = load_dataset(opt, os.path.join(opt.data_path, "result", "black_substitute_valid_0.01.txt"),
                                    os.path.join(opt.data_path, "ptb", "ptb_valid_3.3.0.sd"),
                                    os.path.join(opt.data_path, "result", "black_substitute_valid_0.01.conllx"), train_or_dev='eval')
        opt.test = load_dataset(opt, os.path.join(opt.data_path, "result", "black_substitute_test_0.01.txt"),
                                    os.path.join(opt.data_path, "ptb", "ptb_test_3.3.0.sd"),
                                    os.path.join(opt.data_path, "result", "black_substitute_test_0.01.conllx"), train_or_dev='test')
    #MODIFIED
    #model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    model = get_model(opt, 50265, 50265)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)#, weight_decay=1e-4)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/"%(opt.checkpoint))
    
    #if opt.load_weights is not None and opt.floyd is not None:
    #    os.mkdir('weights')
    #    pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
    #    pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    if opt.is_train:
        train_model(model, opt,'train')
        null(model, opt,'test')
    else:
        predict(model, opt,'test')


def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def promptNextAction(model, opt, SRC, TRG):

    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0
    
    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = 'weights'

    while True:
        save = yesno(input('training complete, save results? [y/n] : '))
        if save == 'y':
            while True:
                if saved_once != 0:
                    res = yesno("save to same folder? [y/n] : ")
                    if res == 'y':
                        break
                dst = input('enter folder name to create for weights (no spaces) : ')
                if ' ' in dst or len(dst) < 1 or len(dst) > 30:
                    dst = input("name must not contain spaces and be between 1 and 30 characters length, enter again : ")
                else:
                    try:
                        os.mkdir(dst)
                    except:
                        res= yesno(input(dst + " already exists, use anyway? [y/n] : "))
                        if res == 'n':
                            continue
                    break
            
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights')
            if saved_once == 0:
                pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
                pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
                saved_once = 1
            
            print("weights and field pickles saved to " + dst)

        res = yesno(input("train for more epochs? [y/n] : "))
        if res == 'y':
            while True:
                epochs = input("type number of epochs to train for : ")
                try:
                    epochs = int(epochs)
                except:
                    print("input not a number")
                    continue
                if epochs < 1:
                    print("epochs must be at least 1")
                    continue
                else:
                    break
            opt.epochs = epochs
            train_model(model, opt)
        else:
            print("exiting program...")
            break

    # for asking about further training use while true loop, and return
if __name__ == "__main__":
    main_for_bi_tir()

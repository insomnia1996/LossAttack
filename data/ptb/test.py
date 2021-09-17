import re,os
with open("./ptb_train_3.3.0.sd",'r') as f:
    lines = [line for line in f if not line.startswith('#')]
    start = 0
    g = open("../result/black_substitute_train_0.01.conllx",'r')
    ori_lines = [line for line in g if not line.startswith('#')]
    ori_sentences = []
    cnt=0
    for i, line in enumerate(ori_lines):
        if len(line) <= 1 and cnt<=100:
            ori_sentence = [l.split()[1] for l in ori_lines[start:i]][:-1]
            ori_sentences.append((" ".join(ori_sentence).lower()))
            cnt+=1
            start = i + 1

    start=0
    cnt=0
    adv_sentences=[]
    for i, line in enumerate(lines):
        if len(line) <= 1 and cnt<=100:
            sentence = [l.split()[1] for l in lines[start:i]]
            adv_sentences.append((" ".join(sentence).lower()))
            cnt+=1
            start = i + 1
    g.close()

    cnt=1
    print(len(ori_sentences),len(adv_sentences))
    h = open("tmp.txt",'w')
    for i in range(len(ori_sentences)):
        print(ori_sentences[i]+'\t'+adv_sentences[i]+'\n')
        h.write(ori_sentences[i]+'\t'+adv_sentences[i]+'\n')
        cnt+=1
        if cnt>100:
            break
    h.close()
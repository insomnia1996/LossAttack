import regex as re
import torch
import os

cnt=[]#index: value 表示第index句在train中是第value句
with open("./data/result/count_map.txt", 'r') as f:
    lines = f.readlines()
    for l in lines:
        cnt.append(int(l.strip()))

def sort(kw):
    with open("./data/result/%s.txt" %kw, 'r') as f:
        g = open("./data/result/%s_sort.txt" %kw, 'w')
        lines=f.readlines()
        sort = [[] for _ in range(len(lines))]
        for i in range(len(lines)):
            line = lines[i].strip()
            sort[cnt[i]]=line
        for j in range(len(lines)):
            g.write(sort[j]+'\n')
        g.close()

kwlst = ['res','gold', 'goldarc', 'goldrel']
for kw in kwlst:
    sort(kw)




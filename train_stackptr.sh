#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python -u train_stackptr.py --mode train --config /home/lyt/LossAttack/LossAttack/models/neuronlp2/models/stackptr.json --num_epochs 100 --batch_size 512 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-6 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.3 --beam 10 \
 --word_embedding glove --word_path "/home/lyt/LossAttack/data/pretrained/GloVe/glove.6B.100d.txt" --char_embedding random \
 --punctuation '.' '``' "''" ':' ',' \
 --train "/home/lyt/LossAttack/data/ptb/ptb_train_3.3.0.sd" \
 --dev "/home/lyt/LossAttack/data/ptb/ptb_valid_3.3.0.sd" \
 --test "/home/lyt/LossAttack/data/ptb/ptb_test_3.3.0.sd" \
 --model_path "data/saved_models/stackptr/"

# LossAttack

Loss Attack is a black-box attack on neural dependency parsers.

# Requirements:

`Python`: 3.7

`PyTorch`: 1.4.0

`transformers`: 4.8.0

# Usage

### Train
To train the target parser, please run:
```sh
$ python run.py -mode=train
```

You can configure the model in `config.ini`

### LossAttack
To attack the trained target parser above, please run:

```sh
$ python run.py -mode=blackbox
```
Make sure the configuration file has the option `blackbox_method` and is set to `substitute`.
* If you want to change the attack corpus, please change the corpus loaded in line 30 in LossAttack/LossAttack/cmds/attack.py to `config.ftrain/fdev/ftest`. Also make sure the configuration `blackbox_index` is set to the correspoding `train/dev/test`.

### Input Reconstruction
To train the Bi-TIR with attacked sentences, please run:
```sh
$ python train.py -epochs 100 -batchsize 300  -is_train
```
Then use the trained model to denoise the adversarial examples using command:
```sh
$ python train.py -k 20 -batchsize 300
$ python dep_val.py
```

After reconstructing the input sentences, attack them again by setting the option `blackbox_method` to `denoise`, then run:
```sh
$ python run.py -mode=blackbox
```


 

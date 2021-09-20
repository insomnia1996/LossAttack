# -*- coding: utf-8 -*-

import argparse
import os
from LossAttack.cmds.DEC.dependency_check import DEC
from LossAttack.cmds import Evaluate, Predict, Train
from LossAttack.cmds.blackbox.blackbox import BlackBoxAttack
from config import Config
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser models.'
    )

    parser.add_argument('--mode', default='hacksubtree',help='running mode choice')
    parser.add_argument('--conf', default='config.ini',help='the path of config file')
    args = parser.parse_args()

    subcommands = {
        'evaluate': Evaluate(),
        'predict': Predict(),
        'train': Train(),
        'blackbox': BlackBoxAttack(),
        'DEC': DEC(),
        # 'subtree': SubTreeAttack(),
        # 'sentencew':WholeSentenceAttack(),
        # 'augmentation':Augmentation()
    }

    print(f"Override the default configs with parsed arguments")
    config = Config(args.conf)
    config.update(vars(args))
    print(config)

    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.device

    print(f"Run the subcommand in mode {config.mode}")
    cmd = subcommands[config.mode]
    cmd(config)

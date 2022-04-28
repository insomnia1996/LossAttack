# -*- coding: utf-8 -*-

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser models.'
    )
    from config import Config
    parser.add_argument('--mode', default='blackbox',help='running mode choice')
    parser.add_argument('--conf', default='config.ini',help='the path of config file')
    args = parser.parse_args()
    print(f"Override the default configs with parsed arguments")
    config = Config(args.conf)
    config.update(vars(args))
    print(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device
    
    
    from LossAttack.cmds.DEC.dependency_check import DEC
    from LossAttack.cmds import Evaluate, Predict, Train
    from LossAttack.cmds.blackbox.blackbox import BlackBoxAttack
    
    import torch
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
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    

    print(f"Run the subcommand in mode {config.mode}")
    cmd = subcommands[config.mode]
    cmd(config)

import logging
import yaml
import argparse
import os
import torch
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

from easydict import EasyDict
from interfaces.Trainer import TextSR
from setup import Logger


def main(config, args):
    Mission = TextSR(config, args)
    Mission.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='super_resolution.yaml')
    parser.add_argument('--rec_backbone', type=str, default='ABINet', choices=["BaseVision","ABINet",'MATRN','PARSeq'])
    parser.add_argument('--go_test', default=False, action='store_true')

    args = parser.parse_args()
    config_path = os.path.join('config', args.config)
    config = yaml.load(open(config_path, 'rb'), Loader=yaml.Loader)
    config = EasyDict(config)
    parser_TPG = argparse.ArgumentParser()

    Logger.init('logs', 'KD-STR', 'train')
    Logger.enable_file()
    main(config, args)

